import horovod.torch as hvd
import math

def adjust_learning_rate(config, nb_iter_per_epoch, epoch, batch_idx, optimizer):
    if config.optim.lr_scheduler.type == "step":
        return adjust_learning_rate_step(config, nb_iter_per_epoch, epoch, batch_idx, optimizer)
    elif config.optim.lr_scheduler.type == "cosine":
        return adjust_learning_rate_cosine(config, nb_iter_per_epoch, epoch, batch_idx, optimizer)
    else:
        raise ValueError(config.optim.lr_scheduler.type)

# Horovod: using `lr = base_lr * hvd.size()` from the very beginning leads to worse final
# accuracy. Scale the learning rate `lr = base_lr` ---> `lr = base_lr * hvd.size()` during
# the first five epochs. See https://arxiv.org/abs/1706.02677 for details.
# After the warmup reduce learning rate by 10 on the 30th, 60th and 80th epochs.
def adjust_learning_rate_step(config, nb_iter_per_epoch, epoch, batch_idx, optimizer):
    K = (hvd.size())
    base_lr = config.optim.base_lr
    base_batch_size = config.optim.base_batch_size
    batch_size = config.optim.train_local_batch_size
    init_lr = base_lr * batch_size / base_batch_size
    if epoch < config.optim.warmup_epochs:
        epoch += float(batch_idx + 1) / nb_iter_per_epoch
        lr_adj = 1. / K * (epoch * (K - 1) / config.optim.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    new_lr = init_lr * K * config.optim.gradient_accumulate * lr_adj
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr

def adjust_learning_rate_cosine(config, nb_iter_per_epoch, epoch, batch_idx, optimizer):
    min_lr = config.optim.min_lr if config.optim.min_lr else 0
    max_epochs = config.optim.epochs
    # K = math.sqrt(hvd.size())
    K = hvd.size()
    base_lr = config.optim.base_lr
    base_batch_size = config.optim.base_batch_size
    batch_size = config.optim.train_local_batch_size
    init_lr = base_lr * batch_size / base_batch_size
    if epoch < config.optim.warmup_epochs:
        epoch += float(batch_idx + 1) / nb_iter_per_epoch
        lr_adj = 1. / K * (epoch * (K - 1) / config.optim.warmup_epochs + 1)
        new_lr = init_lr * K * config.optim.gradient_accumulate * lr_adj
    else:
        Tcur = (epoch - config.optim.warmup_epochs) + (float(batch_idx + 1) / nb_iter_per_epoch )
        Tmax = (max_epochs - config.optim.warmup_epochs)
        max_lr = init_lr * K
        new_lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * (Tcur/Tmax)  ))
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr
