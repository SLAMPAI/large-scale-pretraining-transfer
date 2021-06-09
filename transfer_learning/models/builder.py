import torch
from .bit import KNOWN_MODELS as bit_models

def build_model(config):
    if config.model.type in bit_models:
        model = bit_models[config.model.type](head_size=config.data.nb_classes)
        return model
    else:
        import timm
        model = timm.create_model(config.model.type, num_classes=config.data.nb_classes)
        return model

def load_from_checkpoint(config, path, replace_num_classes_by:int=None):
    if config.model.type in bit_models:
        zero_head = replace_num_classes_by is not None
        model = bit_models[config.model.type](head_size=config.data.nb_classes, zero_head=zero_head)
        weights = torch.load(path, map_location="cpu")["model"]
        model.load_state_dict(weights)
        if replace_num_classes_by:
            fmaps = model.head.conv.weight.shape[1]
            model.head.conv = torch.nn.Conv2d(fmaps, replace_num_classes_by, kernel_size=1, bias=True)
    else:
        import timm
        model = timm.create_model(config.model.type, num_classes=replace_num_classes_by, pretrained=True)
    return model

def extract_features(config, model, batch):
    if config.model.type in bit_models:
        F = model.body(model.root(batch))
        F = torch.nn.AdaptiveAvgPool2d(output_size=1)(F)
        F = F.view(len(F), -1)
        return F
    else:
        raise ValueError(config.model.type)
