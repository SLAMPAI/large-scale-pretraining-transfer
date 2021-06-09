from functools import partial
from torch.optim import SGD
from .sgd_agc import SGD_AGC

def build_optimizer(config, model):
    if config.optim.type == "SGD":
        optimizer = SGD(
            model.parameters(),
            lr=config.optim.base_lr,
            momentum=config.optim.momentum,
            weight_decay=config.optim.weight_decay,
            nesterov=True,
        )
    elif config.optim.type == "SGD_AGC":
        optimizer = SGD_AGC(
            model.parameters(),
            lr=config.optim.base_lr,
            momentum=config.optim.momentum,
            weight_decay=config.optim.weight_decay,
            nesterov=True,
            output_size=config.data.nb_classes,
        )
    else:
        raise ValueError(config.optim.type)
    return optimizer
