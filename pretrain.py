from functools import partial
import math
import os
import datetime
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter

import horovod.torch as hvd

from transfer_learning.dataloaders.builder import build_dataloader
from transfer_learning.optim.builder import build_optimizer
from transfer_learning.models.builder import build_model
from transfer_learning.lr_scheduler.builder import adjust_learning_rate

try:
    #From: https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    from apex import amp
except ImportError:
    # Apex (for mixed precision) is not mandatory, as the native mixed precision of PyTorch can be used
    pass

def train(epoch):
    global nb_images
    model.train()
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')
    if config.mixed_precision.enabled:
        if config.mixed_precision.backend == "native":
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            autocast = torch.cuda.amp.autocast
        elif config.mixed_precision.backend == "apex":
            autocast = DummyAutoCast
    else:
        autocast = DummyAutoCast
    local_batch_size = config.optim.train_local_batch_size

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            nb_images += len(data)
            new_lr = adjust_learning_rate(config, len(train_loader), epoch, batch_idx, optimizer)
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), local_batch_size):
                with autocast():#for mixed precision (in case it is used)
                    data_batch = data[i:i + local_batch_size]
                    target_batch = target[i:i + local_batch_size]
                    output = model(data_batch)
                    loss = compute_loss(output, target_batch)
                    acc = accuracy(output, target_batch)
                    train_accuracy.update(acc)
                    loss.div_(math.ceil(float(len(data)) / local_batch_size))
                    if config.mixed_precision.enabled:
                        if config.mixed_precision.backend == "apex":
                            #https://github.com/NVIDIA/apex/tree/master/examples/imagenet#mixed-precision-imagenet-training-in-pytorch
                            with amp.scale_loss(loss, optimizer) as scaled_loss:
                                #https://github.com/NVIDIA/apex/issues/307
                                #https://gist.github.com/alsrgv/0713add50fe49a409316832a31612dde
                                scaled_loss.backward()
                                optimizer.synchronize()
                            with optimizer.skip_synchronize():
                                optimizer.step()
                        elif config.mixed_precision.backend == "native":
                            scaler.scale(loss).backward()
                            optimizer.synchronize()
                            with optimizer.skip_synchronize():
                                scaler.step(optimizer)
                                scaler.update()
                    else:
                        # Average gradients among sub-batches
                        loss.backward()
                        optimizer.step()
                    train_loss.update(loss)
            # Gradient is applied across all ranks
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)
        log_writer.add_scalar('learning_rate', new_lr, epoch)

def compute_loss(output, target_batch):
    if target_batch.ndim == 2:
        #multi-label setting
        mask = ~torch.isnan(target_batch)
        output = output[mask]
        target_batch = target_batch[mask]
        loss = F.binary_cross_entropy_with_logits(output, target_batch)
    else:
        #single-label multi-class setting
        loss = F.cross_entropy(output, target_batch)
    return loss

def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(compute_loss(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy', val_accuracy.avg, epoch)
    return val_loss.avg.item(), val_accuracy.avg.item()

class DummyAutoCast():
    # dummy autocast class for single precision
    def __enter__(self):
        pass
    def __exit__(self, *args, **kwargs):
        pass


def accuracy(output, target):
    if target.ndim == 1:
        #single label multi-class setting
        # get the index of the max log-probability
        pred = output.max(1, keepdim=True)[1]
        return pred.eq(target.view_as(pred)).cpu().float().mean()
    elif target.ndim == 2:
        #multi-label setting
        mask = ~torch.isnan(target) # in medical data, some targets are nan, ignore them
        #output is expected to be logits, so apply sigmoid and threshold at 0.5 by default
        pred = ((output[mask].sigmoid()) > 0.5).float()
        target = target[mask]
        return (pred==target).float().mean()
    else:
        raise ValueError("target.ndim not 1 or 2")

def save_best(epoch, log_dir):
    filepath = os.path.join(log_dir, "model.pth.tar")
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, filepath)

def save_checkpoint(epoch, log_dir):
    if hvd.rank() == 0:
        filepath = os.path.join(log_dir, config.logging.checkpoint_format.format(epoch=epoch + 1))
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        mean = hvd.allreduce(val.detach().cpu(), name=self.name)
        self.sum += mean
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


if __name__ == '__main__':
    from omegaconf import OmegaConf
    import argparse
    import shutil
    parser = argparse.ArgumentParser(
        description='PyTorch pre-training script',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config-file', default="config_example.yaml", type=str, required=True)
    parser.add_argument('--log-dir', default=None, type=str, required=False)
    parser.add_argument('--resume', action='store_true',  default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    args = parser.parse_args()
    config = OmegaConf.load(args.config_file)
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    hvd.init()
    torch.manual_seed(config.seed)
    if hvd.rank() == 0:
        print(f"Number of workers: {hvd.size()}")
        print(f"Effective batch size: {config.optim.train_local_batch_size * config.optim.gradient_accumulate * hvd.size()}")
    
    # logs_dir is the logging directory
    # if provided in args or config file, use it.
    # if not, use the directory logs/<model_name> as the default
    # where <model_name> is extracted from the config file name
    if args.log_dir:
        log_dir = args.log_dir
    elif config.logging.log_dir:
        log_dir = config.logging.log_dir
    else:
        name = os.path.basename(args.config_file).split(".")[0]
        log_dir = os.path.join("logs", name)
    os.makedirs(log_dir, exist_ok=True)

    # for reproduction, put the confg file in logging directory
    shutil.copy(args.config_file, os.path.join(log_dir, "config.yaml"))
    if not config.logging.checkpoint_format:
        config.logging.checkpoint_format = "./checkpoint-{epoch}.pth.tar"

    if config.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(config.seed)

    cudnn.benchmark = True
    cudnn.enabled = True

    # If set > 0, will resume training from a given checkpoint.
    resume_from_epoch = 0
    if hvd.rank() == 0 and args.resume:
        for try_epoch in range(config.optim.epochs, 0, -1):
            if os.path.exists(os.path.join(log_dir, config.logging.checkpoint_format.format(epoch=try_epoch))):
                resume_from_epoch = try_epoch
                break

    # Horovod: broadcast resume_from_epoch from rank 0 (which will have
    # checkpoints) to other ranks.
    resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch), root_rank=0,
                                      name='resume_from_epoch').item()
    if hvd.rank() == 0 and resume_from_epoch:
        print(f"Resume from epoch: {resume_from_epoch}")
    # Horovod: print logs on the first worker.
    verbose = 1 if hvd.rank() == 0 else 0

    # Horovod: write TensorBoard logs on first worker.
    if hvd.rank() == 0:
        print("Log Dir", log_dir)
    log_writer = SummaryWriter(log_dir) if hvd.rank() == 0 else None

    train_loader, val_loader = build_dataloader(config)

    model = build_model(config)
    optimizer = build_optimizer(config, model)
    if args.cuda:
        # Move model to GPU.
        model.cuda()

    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if config.horovod.fp16_allreduce else hvd.Compression.none

    # Horovod: wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters(),
        compression=compression,
        backward_passes_per_step=config.optim.gradient_accumulate,
        op=hvd.Average,
    )

    # Restore from a previous checkpoint, if initial_epoch is specified.
    # Horovod: restore on the first worker which will broadcast weights to other workers.
    if (resume_from_epoch > 0) and args.resume and hvd.rank() == 0:
        filepath = os.path.join(log_dir, config.logging.checkpoint_format.format(epoch=resume_from_epoch))
        print(f"Resume from {filepath}")
        checkpoint = torch.load(filepath, map_location="cpu")
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Horovod: broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    if config.mixed_precision.enabled and config.mixed_precision.backend == "apex":
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")

    # used for computing throughput
    nb_images = 0
    start = time.time()
    

    # load the current best val loss
    # if it does not exist, put it to inf by default
    best_val_loss_path = os.path.join(log_dir,"best_val_loss")
    if os.path.exists(best_val_loss_path):
        best_val_loss = float(open(best_val_loss_path).read())
    else:
        best_val_loss = float("inf")

    for epoch in range(resume_from_epoch, config.optim.epochs):
        if hasattr(train_loader, "sampler"):
            # for sharding
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
        # train for one epoch
        train(epoch)
        if config.logging.validate:
            val_loss, val_acc = validate(epoch)
            if hvd.rank() == 0 and (val_loss < best_val_loss):
                # save the best model
                print(f"Improved val loss from {best_val_loss} to {val_loss}")
                best_val_loss = val_loss
                with open(best_val_loss_path, "w") as fd:
                    fd.write(str(best_val_loss))
                save_best(epoch, log_dir)
        if config.logging.save_checkpoint:
            # save the model at each epoch
            save_checkpoint(epoch, log_dir)
    if hvd.rank() == 0:
        # show throughput
        duration = time.time() - start
        nb_images_processed = nb_images* hvd.size()
        print(f"total images: {nb_images_processed}")
        print(f"total training time in sec: {duration}")
        print(f"total images/sec/gpu: {(nb_images_processed/duration)/hvd.size()}")
        print(f"total images/sec: {nb_images_processed/duration}")
