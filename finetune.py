from functools import partial
import math
import os
import datetime
import time
import argparse
from tqdm import tqdm
import torch.multiprocessing as mp # if not used, "memory mapped error" in the dataloader, no idea why
import os
from os.path import join as pjoin  # pylint: disable=g-importing-member
from copy import deepcopy
import time
import pandas as pd
import random
import numpy as np
import torch
import torchvision as tv
import torch.nn.functional as F
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from transfer_learning.finetuning import fewshot as fs
from transfer_learning.finetuning import lbtoolbox as lb
from transfer_learning.finetuning import bit_common
from transfer_learning.finetuning import bit_hyperrule

from transfer_learning.models import bit as models

from transfer_learning.optim.sgd_agc import SGD_AGC
from transfer_learning.models.builder import load_from_checkpoint
from transfer_learning.datasets.builder import build_dataset
from transfer_learning.optim.builder import build_optimizer
from transfer_learning.dataloaders.builder import build_dataloader


from omegaconf import OmegaConf

try:
    import horovod.torch as hvd
    USE_HOROVOD = True
except ImportError:
    USE_HOROVOD = False

MULTIPROCESSING_CONTEXT = "forkserver"

# if USE_HOROVOD:
    # from transfer_learning.dataloaders.dataloader_threads import DataLoader
# else:

from torch.utils.data import DataLoader

def topk(output, target, ks=(1,)):
    """Returns one boolean vector for each k, whether the target is within the output's top-k."""
    nb_classes = output.shape[1]
    ks = [min(k, nb_classes) for k in ks]
    _, pred = output.topk(max(ks), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [correct[:k].max(0)[0] for k in ks]


def recycle(iterable):
    """Variant of itertools.cycle that does not save iterates."""
    while True:
        for i in iterable:
            yield i


def mktrainval(args, config, logger):
    """Returns train and validation datasets."""
    train_set, valid_set = build_dataset(config)
    if args.valid_ratio is not None:
        orig_train = deepcopy(train_set)
        rng = np.random.RandomState(args.valid_seed)
        inds = np.arange(len(train_set))
        rng.shuffle(inds)
        valid_set_ = deepcopy(train_set)
        if hasattr(valid_set, "transform"):
            valid_set_.transform = valid_set.transform
        if hasattr(valid_set, "data_aug"):
            valid_set_.data_aug = valid_set.data_aug
        nb_train = int(len(train_set) * (1 - args.valid_ratio))
        train_set = torch.utils.data.Subset(train_set, inds[0:nb_train])
        train_set.labels = orig_train.labels[inds[0:nb_train]]
        valid_set = torch.utils.data.Subset(valid_set_, inds[nb_train:])
        valid_set.labels = orig_train.labels[inds[nb_train:]]
    if args.multilabel_force_classification:
        assert hasattr(train_set, "labels")
        assert hasattr(valid_set, "labels")
        train_set = SingleLabelFromMultiLabel(train_set, train_set.labels)
        valid_set = SingleLabelFromMultiLabel(valid_set, valid_set.labels)

    if args.train_ratio is not None:
        logger.info(f"Train ratio: {args.train_ratio}")
        rng = np.random.RandomState(args.seed)
        indices = np.arange(len(train_set))
        rng.shuffle(indices)
        train_size = int(len(train_set) * args.train_ratio)
        indices = indices[:train_size]  
        train_set = torch.utils.data.Subset(train_set, indices)

    if config.data.oversampling == True and args.examples_per_class is None:
        logger.info('Using oversampling to deal with class imbalance')
        loader = DataLoader(
            train_set,
            shuffle=False,
            num_workers=config.data.workers,
            batch_size=config.optim.batch,
            worker_init_fn=seed_worker,
            multiprocessing_context=MULTIPROCESSING_CONTEXT,

        )
        #Thanks to https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/2
        class_weight = torch.zeros(config.data.nb_classes)
        freq = [0] * config.data.nb_classes
        ys = []
        for x, y in loader:
            for yi in y:
                class_weight[yi] += 1
                ys.append(yi)
        nb_examples = len(ys)
        class_weight = nb_examples  / class_weight
        logger.info(f'Class weight for oversampling: {class_weight}')
        weight = [0] * nb_examples
        for i, yi in enumerate(ys):
            weight[i] = class_weight[yi]
        weight = torch.FloatTensor(weight)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(weight, len(weight))
    else:
        if USE_HOROVOD:
            sampler = torch.utils.data.DistributedSampler(
                train_set,
                num_replicas=hvd.size(),
                rank=hvd.rank(),
            )
        else:
            sampler = None
    if args.examples_per_class is not None:
        logger.info(f"Looking for {args.examples_per_class} images per class...")
        if hasattr(train_set, "labels"):
            rng = np.random.RandomState(args.seed)
            inds = np.arange(len(train_set))
            indices = []
            for cl in range(len(train_set.classes)):
                inds_cl = inds[train_set.labels==cl]
                rng.shuffle(inds_cl)
                inds_cl = inds_cl[0:args.examples_per_class]
                indices.append(inds_cl)
            indices = np.concatenate(indices)
            rng.shuffle(indices)
            # print(indices)
        else:
            indices = fs.find_fewshot_indices(train_set, args.examples_per_class, random_state=args.seed)
        train_set = torch.utils.data.Subset(train_set, indices=indices)
        #DEBUG
        # import torchvision
        # for i in range(len(train_set)):
            # x, y = train_set[i]
            # x = (x-x.min())/(x.max()-x.min())
            # print(i, x.shape)
            # torchvision.utils.save_image(x, f"{i}.jpg")
    # for i in range(len(train_set)):
        # print(i, train_set[i][1])
    logger.info(f"Using a training set with {len(train_set)} images.")
    logger.info(f"Using a validation set with {len(valid_set)} images.")

    micro_batch_size = config.optim.batch // config.optim.batch_split

    valid_loader = DataLoader(
        valid_set,
        batch_size=config.optim.val_batch_size if config.optim.val_batch_size else micro_batch_size,
        shuffle=False,
        num_workers=config.data.workers,
        # pin_memory=True,
        drop_last=False,
        worker_init_fn=seed_worker,
        multiprocessing_context=MULTIPROCESSING_CONTEXT,
    )

    if micro_batch_size <= len(train_set):
        train_loader = DataLoader(
            train_set,
            batch_size=micro_batch_size,
            shuffle=True if sampler is None else None,
            num_workers=config.data.workers,
            # pin_memory=True,
            drop_last=False,
            sampler=sampler,
            worker_init_fn=seed_worker,
            multiprocessing_context=MULTIPROCESSING_CONTEXT,
        )
    else:
        # In the few-shot cases, the total dataset size might be smaller than the batch-size.
        # In these cases, the default sampler doesn't repeat, so we need to make it do that
        # if we want to match the behaviour from the paper.
        train_loader = DataLoader(
            train_set,
            batch_size=micro_batch_size,
            num_workers=config.data.workers,
            # pin_memory=True,
            sampler=torch.utils.data.RandomSampler(
                train_set, replacement=True, num_samples=micro_batch_size
            ),
            worker_init_fn=seed_worker,
            multiprocessing_context=MULTIPROCESSING_CONTEXT,
        )
    # train_loader.sampler = sampler
    return train_set, valid_set, train_loader, valid_loader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SingleLabelFromMultiLabel:

    def __init__(self, dataset, labels):
        self.dataset = dataset
        labels = np.copy(labels)
        labels[np.isnan(labels)] = 0
        # only keep examples where there is exactly one class present
        mask = (labels.sum(axis=1)==1)
        inds = np.arange(len(labels))
        inds = inds[mask]
        self.labels = labels.argmax(axis=1)[inds]
        self.subset = torch.utils.data.Subset(dataset, inds)
        self.classes = range(labels.shape[1])

    def __getitem__(self, idx):
        x, yi = self.subset[idx]
        y = self.labels[idx]
        # print(yi, yi.argmax(), y)
        # assert yi.argmax() == y
        return x, y

    def __len__(self):
        return len(self.subset)


def run_eval(model, data_loader, device, chrono, logger, step, task):
    # switch to evaluate mode
    model.eval()

    logger.info("Running validation...")
    logger.flush()

    all_c, all_top1, all_top5, all_true, all_pred_proba = [], [], [], [], []
    end = time.time()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # measure data loading time
            chrono._done("eval load", time.time() - end)

            # compute output, measure accuracy and record loss.
            with chrono.measure("eval fprop"):
                logits = model(x)
                if task == "classification":
                    c = torch.nn.CrossEntropyLoss(reduction="none")(logits, y)
                    top1, top5 = topk(logits, y, ks=(1, 5))
                    all_top1.extend(top1.cpu())
                    all_top5.extend(top5.cpu())
                    all_c.extend(c.cpu())  # Also ensures a sync point.
                    all_true.append(y.cpu().numpy())
                    all_pred_proba.append(logits.softmax(dim=-1).cpu().numpy())
                elif task == "multilabel":
                    all_true.append(y.cpu().numpy())
                    all_pred_proba.append(logits.sigmoid().cpu().numpy())
                    mask = ~torch.isnan(y)
                    logits = logits[mask]
                    y = y[mask]
                    c = F.binary_cross_entropy_with_logits(logits, y).item()
                    all_c.append(c)  # Also ensures a sync point.

        # measure elapsed time
        end = time.time()

    model.train()
    logger.info(
        f"Validation@{step} loss {np.mean(all_c):.5f}, "
    )
    if all_top1 and all_top5:
        logger.info(
            f"top1 {np.mean(all_top1):.2%}, "
            f"top5 {np.mean(all_top5):.2%}"
        )
    
    logger.flush()
    
    all_true = np.concatenate(all_true)
    all_pred_proba = np.concatenate(all_pred_proba)
    return all_c, all_top1, all_top5, all_true, all_pred_proba


def mixup_data(x, y, l):
    """Returns mixed inputs, pairs of targets, and lambda"""
    indices = torch.randperm(x.shape[0]).to(x.device)

    mixed_x = l * x + (1 - l) * x[indices]
    y_a, y_b = y, y[indices]
    return mixed_x, y_a, y_b


def mixup_criterion(criterion, pred, y_a, y_b, l):
    return l * criterion(pred, y_a) + (1 - l) * criterion(pred, y_b)

def select_finetuning_options(cfg, pretrain_name):
    for k, v in cfg.items():
        keys = k.split(",")
        if pretrain_name in keys:
            print(f"Using finetuning configuration from {pretrain_name}")
            return v
    return cfg


def main(args):
    if USE_HOROVOD:
        hvd.init()
     
    if USE_HOROVOD and hvd.rank() > 0:
        logger = logging.getLogger('NullLogger')
    else:
        logger = bit_common.setup_logger(args)

    pretrain_config = OmegaConf.load(args.pretrain_config_file)
    finetune_config = OmegaConf.load(args.finetune_config_file)
    if pretrain_config.data.mean:
        finetune_config.data.mean = pretrain_config.data.mean
    if pretrain_config.data.std:
        finetune_config.data.std = pretrain_config.data.std
    # name = os.path.splitext(os.path.basename(args.pretrain_config_file))[0]
    # finetune_config = select_finetuning_options(finetune_config, name)

    if args.seed is not None:
        from torch.backends import cudnn
        from glob import glob
        nb_runs = len(glob(os.path.join(args.logdir, "events*")))
        global_seed = args.seed + nb_runs*100
        logger.info(f"Using global seed of {args.seed} + {nb_runs}*100")
        rng = random.Random(global_seed)
        if USE_HOROVOD:
            seed = hash((global_seed, hvd.rank())) % (2 ** 32)
        else:
            seed = global_seed
        # follow https://pytorch.org/docs/stable/notes/randomness.html
        # for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        # torch.use_deterministic_algorithms(True)
        #https://github.com/pytorch/pytorch/issues/38410
        cudnn.deterministic = True # type: ignore
        cudnn.benchmark = False # type: ignore
    
    if args.batch_split is not None:
        # override batch_split if
        finetune_config.optim.batch_split = args.batch_split

    if USE_HOROVOD:
        finetune_config.optim.batch //= hvd.size()
        finetune_config.optim.batch_split //= hvd.size()
        logger.info(f"Number of GPU workers: {hvd.size()}")
        logger.info(f"GPU local batch size: {finetune_config.optim.batch}")
        logger.info(f"Batch Split: {finetune_config.optim.batch_split}")

    # Lets cuDNN benchmark conv implementations and choose the fastest.
    # Only good if sizes stay the same within the main loop!
    torch.backends.cudnn.benchmark = True
    if USE_HOROVOD:
        torch.cuda.set_device(hvd.local_rank())
        # torch.cuda.set_device(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Going to train on {device}")
    
    if USE_HOROVOD and hvd.rank() > 0:
        log_writer = None
    else:
        log_writer = SummaryWriter(os.path.join(args.logdir))

    train_set, valid_set, train_loader, valid_loader = mktrainval(args, finetune_config, logger)
    nb_classes = finetune_config.data.nb_classes
    
    if args.model_path is None:
        name, *rest = os.path.splitext(os.path.basename(args.pretrain_config_file))
        path = os.path.join("pretrained_models", name, "model.pth.tar")
        assert os.path.exists(path), f"Model checkpoint does not exist: {path}"
        args.model_path = path
    logger.info(f"Using pre-trained model from {args.model_path}")
    model = load_from_checkpoint(pretrain_config, args.model_path, replace_num_classes_by=nb_classes)
    logger.info("Moving model onto all GPUs")
    step = 0
    model = model.to(device)
    
    optim = build_optimizer(finetune_config, model)
    savename = pjoin(args.logdir, "model.pth.tar")
    try:
        logger.info(f"Model will be saved in '{savename}'")
        checkpoint = torch.load(savename, map_location="cpu")
        logger.info(f"Found saved model to resume from at '{savename}'")

        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model"])
        optim.load_state_dict(checkpoint["optim"])
        logger.info(f"Resumed at step {step}")
    except FileNotFoundError:
        logger.info("Fine-tuning")
     
    if USE_HOROVOD:
        # compression = hvd.Compression.fp16 if finetune_config.horovod.fp16_allreduce else hvd.Compression.none
        compression = hvd.Compression.none
        optim = hvd.DistributedOptimizer(
            optim, named_parameters=model.named_parameters(),
            compression=compression,
            backward_passes_per_step=finetune_config.optim.batch_split,
            op=hvd.Average,
        )
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optim, root_rank=0)

    optim.zero_grad()
    
    if args.steps:
        finetune_config.optim.steps = args.steps

    if finetune_config.optim.steps:
        bit_hyperrule_train_set = bit_hyperrule.get_max_train_set(finetune_config.optim.steps)
    else:
        bit_hyperrule_train_set = len(train_set)
    logger.info(f"Bit HyperRule schedule: {bit_hyperrule.get_schedule(bit_hyperrule_train_set)}")
    model.train()
    mixup = bit_hyperrule.get_mixup(bit_hyperrule_train_set)
    if finetune_config.optim.class_weight == 'balanced' and args.examples_per_class is None:
        class_weight = torch.zeros(nb_classes)
        freq = [0] * nb_classes
        for x, y in train_loader:
            for yi in y:
                class_weight[yi] += 1
        # minority class: weight of 1
        class_weight = torch.min(class_weight) / class_weight
        logger.info(f'Imbalanced data, using the following class weights: {class_weight}')
    else:
        class_weight = None
    
    x, y = train_set[0]
    if type(y) != int and len(y.shape) == 1:
        cri = torch.nn.BCEWithLogitsLoss(weight=class_weight).to(device)
        def cri(output, target_batch):
            mask = ~torch.isnan(target_batch)
            output = output[mask]
            target_batch = target_batch[mask]
            loss = F.binary_cross_entropy_with_logits(output, target_batch)
            return loss
        task = "multilabel"
    else:
        cri = torch.nn.CrossEntropyLoss(weight=class_weight).to(device)
        task = "classification"
    logger.info(task)
    logger.info("Starting training!")
    chrono = lb.Chrono()
    accum_steps = 0
    mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1
    end = time.time()
    
    steps_per_epoch = len(train_set) // finetune_config.optim.batch
    for (x, y) in recycle(train_loader):
        # if step == 110:
            # break
        if USE_HOROVOD and hasattr(train_loader.sampler, 'set_epoch'):
            epoch = step // steps_per_epoch
            new_epoch = (step % steps_per_epoch) == 0
            if new_epoch:
                train_loader.sampler.set_epoch(epoch)
        # measure data loading time, which is spent in the `for` statement.
        chrono._done("load", time.time() - end)

        # Schedule sending to GPU(s)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        # Update learning-rate, including stop training if over.
        lr = bit_hyperrule.get_lr(step, bit_hyperrule_train_set, finetune_config.optim.base_lr)
        if lr is None:
            # lr is None means last step of the bit hyperrule schedule was achieved
            # in original bit hyperrule, this is where we would stop, but we can imagine
            # cases where we want to continue training for a little bit, this can be done
            # using the config param `steps`
            steps = finetune_config.optim.steps
            # if `steps` is not provided in the config file, stop training
            if steps is None:
                break
            # if `steps` is provided in config file, check if we achieved `steps` number
            # of steps, if so, stop training
            if step >= steps:
                break
            # use the last learning rate from bit hyperrule learning rate schedule
            lr = bit_hyperrule.get_final_lr(bit_hyperrule_train_set, base_lr=finetune_config.optim.base_lr)
        for param_group in optim.param_groups:
            param_group["lr"] = lr
        if mixup > 0.0:
            x, y_a, y_b = mixup_data(x, y, mixup_l)
        # compute output
        with chrono.measure("fprop"):
            logits = model(x)
            if mixup > 0.0:
                c = mixup_criterion(cri, logits, y_a, y_b, mixup_l)
            else:
                c = cri(logits, y)
            c_num = float(c.data.cpu().numpy())  # Also ensures a sync point.

        # Accumulate grads
        with chrono.measure("grads"):
            (c / finetune_config.optim.batch_split).backward()
            accum_steps += 1

        accstep = (
            f" ({accum_steps}/{finetune_config.optim.batch_split})" if finetune_config.optim.batch_split > 1 else ""
        )
        if step % 10 == 0:
            logger.info(
                f"[step {step}{accstep}]: loss={c_num:.5f} (lr={lr:.1e})"
            )  # pylint: disable=logging-format-interpolation
            if log_writer:
                log_writer.add_scalar('train/loss', c_num, step)
        if hasattr(logger, "flush"):
            logger.flush()

        # Update params
        if accum_steps == finetune_config.optim.batch_split:
            with chrono.measure("update"):
                # for param in model.parameters():
                    # print(torch.norm(param), torch.norm(param.grad))
                optim.step()
                optim.zero_grad()
            step += 1
            accum_steps = 0
            # Sample new mixup ratio for next batch
            mixup_l = np.random.beta(mixup, mixup) if mixup > 0 else 1

            # Run evaluation and save the model.
            if (finetune_config.logging.eval_every and step % finetune_config.logging.eval_every == 0):

                if ((USE_HOROVOD and hvd.rank() == 0) or not USE_HOROVOD):
                    logger.info(f"Timings:\n{chrono}")
                    all_c, all_top1, all_top5, y_true, y_pred_proba = run_eval(model, valid_loader, device, chrono, logger, step, task)
                    log_eval(logger, log_writer, all_c, y_true, y_pred_proba, step)
                    if args.save:
                        torch.save(
                            {
                                "step": step,
                                "model": model.state_dict(),
                                "optim": optim.state_dict(),
                                "pretrain_config": pretrain_config,
                                "finetune_config": finetune_config,
                            },
                            savename,
                        )
                if USE_HOROVOD:
                    hvd.join()

        end = time.time()

    # Final eval at end of training.
    # all_c, all_top1, all_top5 = run_eval(model, valid_loader, device, chrono, logger, step="end")
    if (((USE_HOROVOD and hvd.rank() == 0) or not USE_HOROVOD)):
        all_c, all_top1, all_top5, y_true, y_pred_proba = run_eval(model, valid_loader, device, chrono, logger, step, task)
        log_eval(logger, log_writer, all_c, y_true, y_pred_proba, step)
    # log_writer.add_scalar('test/loss', np.mean(all_c), step)
    # log_writer.add_scalar('test/acc', np.mean(all_top1), step)
    logger.info(f"Timings:\n{chrono}")
    logger.info("Training Finished Successfully")


def log_eval(logger, log_writer, all_loss, y_true, y_pred_proba, step):
    multilabel = len(y_true.shape) == 2

    if multilabel:
        y_pred = (y_pred_proba>0.5)
    else:
        y_pred = y_pred_proba.argmax(axis=1)
        logger.info(classification_report(y_true, y_pred))
    auc_vals = []
    avg_prec_vals = []
    precisions = []
    recalls = []
    f1s = []
    supports = []
    supports_pos = []
    nb_classes = y_pred_proba.shape[1]
    for class_id in range(nb_classes):
        
        if multilabel:
            yt = y_true[:, class_id]
            yp = y_pred_proba[:, class_id] > 0.5
            y_pr = y_pred_proba[:, class_id]
            mask = ~np.isnan(yt)
            yt = yt[mask]
            yp = yp[mask]
            y_pr = y_pr[mask]
        else:
            yt = (y_true == class_id)
            yp = (y_pred == class_id)
            y_pr = y_pred_proba[:, class_id]
        support = len(yt)
        support_pos = int(yt.sum())
        precision = precision_score(yt, yp)
        recall = recall_score(yt, yp)
        f1 = f1_score(yt, yp)
        try:
            avg_prec_val = average_precision_score(yt, y_pr, pos_label=1)
        except ValueError:
            avg_prec_val = np.nan
        try:
            auc_val = roc_auc_score(yt, y_pr)
        except ValueError:
            # happens when there is one class only
            auc_val = np.nan
        logger.info(f"Class {class_id} average precision: {avg_prec_val:.3f}")
        logger.info(f"Class {class_id} AUC: {auc_val:.3f}")
        logger.info(f"Class {class_id} Precision: {precision:.3f}")
        logger.info(f"Class {class_id} Recall: {recall:.3f}")
        logger.info(f"Class {class_id} F1: {f1:.3f}")
        if log_writer:
            log_writer.add_scalar(f'test/average_precision_class_{class_id}', avg_prec_val, step)
            log_writer.add_scalar(f'test/AUC_class_{class_id}', auc_val, step)
            log_writer.add_scalar(f'test/precision_{class_id}', precision, step)
            log_writer.add_scalar(f'test/recall_{class_id}', recall, step)
            log_writer.add_scalar(f'test/f1_{class_id}', f1, step)
        auc_vals.append(auc_val)
        avg_prec_vals.append(avg_prec_val)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)
        supports_pos.append(support_pos)
    supports_neg = np.array(supports) - np.array(supports_pos)
    report = pd.DataFrame({
        "class": np.arange(nb_classes),
        "precision": precisions,
        "recall": recalls,
        "f1": f1s,
        "AUC": auc_vals,
        "AvgPrecision": avg_prec_vals,
        # "support": supports,
        "support_pos": supports_pos,
        "support_neg": supports_neg,
    })
    # print(report)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    logger.info("\n" + str(report))
    mean_average_precision = np.nanmean(avg_prec_vals)
    mean_auc = np.nanmean(auc_vals)
    logger.info(f"Mean Average Precision: {mean_average_precision:.3f}")
    logger.info(f"Mean AUC: {mean_auc:.3f}")
    
    if log_writer:
        log_writer.add_scalar('test/mean_average_precision', mean_average_precision, step)
        log_writer.add_scalar('test/mean_auc', mean_auc, step)
    acc = (y_pred == y_true).mean()
    logger.info(f"ACC: {acc:.3f}")
    if log_writer:
        log_writer.add_scalar('test/acc', acc, step)
    if all_loss is not None and log_writer:
        log_writer.add_scalar('test/loss', np.mean(all_loss), step)
    return log_writer

if __name__ == "__main__":
    parser = bit_common.argparser(models.KNOWN_MODELS.keys())
    parser.add_argument('--pretrain-config-file', default="config_example.yaml", type=str, required=True)
    parser.add_argument('--finetune-config-file', default="config_example.yaml", type=str, required=True)
    parser.add_argument('--seed', default=None, type=int, required=False)
    parser.add_argument('--batch-split', default=None, type=int, required=False)
    parser.add_argument("--model-path", required=False, help="Path to weights")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--train-ratio", default=None, type=float)
    parser.add_argument("--multilabel-force-classification", default=False, action="store_true")
    parser.add_argument("--valid-ratio", default=None, type=float)
    parser.add_argument("--valid-seed", default=0, type=int)
    parser.add_argument("--steps", default=None, type=int)
    main(parser.parse_args())
