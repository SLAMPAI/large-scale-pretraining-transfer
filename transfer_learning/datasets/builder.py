try:
    import horovod.torch as hvd
    HOROVOD = True
except Exception:
    HOROVOD = False
from copy import deepcopy
import numpy as np
import torchvision.transforms as transforms
import torchvision

from .caffe_lmdb import CaffeLMDB
from .tuberculosis import Tuberculosis
from .pets import Pets
from . import chexpert
from . import mimic
from . import image_folder_numpy
from . import covidx_cxr2
from . import nih
from . import padchest

def build_dataset(config):
    train_transform, val_transform = build_transforms(config)
    train_dir = config.data.train_dir
    val_dir = config.data.val_dir
    dataset_type = config.data.type

    if dataset_type == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10("cifar10", train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10("cifar10", train=False, download=True, transform=val_transform)
        train_dataset.nb_classes = 10
    elif dataset_type == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100("cifar100", train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100("cifar100", train=False, download=True, transform=val_transform)
        train_dataset.nb_classes = 100
        train_dataset.nb_classes = 100
    elif dataset_type == "tuberculosis":
        if "," in train_dir:
            train_dirs = train_dir.split(",")
            val_dirs = val_dir.split(",")
            train_dataset = Merge([Tuberculosis(train_dir, train=True, transform=train_transform) for train_dir in train_dirs])
            val_dataset = Merge([Tuberculosis(val_dir, train=False, transform=val_transform) for val_dir in val_dirs])
        else:
            train_dataset = Tuberculosis(train_dir, train=True, transform=train_transform)
            val_dataset = Tuberculosis(val_dir, train=False, transform=val_transform)
    elif dataset_type == "lmdb":
        train_dataset = CaffeLMDB(
            train_dir,
            transform=train_transform
        )
        val_dataset = CaffeLMDB(
            val_dir,
            transform=val_transform
        )
    elif dataset_type == "image_folder":
        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=val_transform)
    elif dataset_type == "pets":
        train_dataset = Pets(train_dir, train=True, transform=train_transform)
        val_dataset = Pets(val_dir, train=False, transform=val_transform)
    elif dataset_type == "covidx_cxr2":
        train_dataset = covidx_cxr2.COVIDX_CXR2(train_dir, train=True, transform=train_transform)
        val_dataset = covidx_cxr2.COVIDX_CXR2(val_dir, train=False, transform=val_transform)
    elif dataset_type == "image_folder_numpy":
        train_dataset = image_folder_numpy.ImageFolderNumpy(train_dir, transform=train_transform)
        val_dataset = image_folder_numpy.ImageFolderNumpy(val_dir, transform=val_transform)
    elif dataset_type in ("chexpert", "mimic", "nih", "padchest"):
        aug = None
        if config.data.augmentation.aug == "cohen_chexpert_aug":
            aug = chexpert.cohen_aug

        if config.data.unique_patients:
            unique_patients = config.data.unique.unique_patients
        else:
            unique_patients = False

        if config.data.views:
            views = list(map(str, config.data.views))
        else:
            views = ["PA", "AP"]
        if dataset_type == "chexpert":
            train_dataset = chexpert.ChexPert(train_dir, split="train", transform=train_transform, aug=aug, views=views, unique_patients=unique_patients)
            val_dataset = chexpert.ChexPert(train_dir, split="valid", transform=val_transform, views=views, unique_patients=unique_patients)
        elif dataset_type == "mimic":
            train_dataset = mimic.MIMIC(train_dir, split="train", transform=train_transform, aug=aug, views=views, unique_patients=unique_patients)
            val_dataset = mimic.MIMIC(train_dir, split="validate", transform=val_transform, views=views, unique_patients=unique_patients)
        elif dataset_type == "nih":
            train_dataset = nih.NIH(train_dir, train=True, transform=train_transform, aug=aug, views=views, unique_patients=unique_patients)
            val_dataset = nih.NIH(train_dir, train=False, transform=val_transform, views=views, unique_patients=unique_patients)
        elif dataset_type == "padchest":
            train_dataset = padchest.PadChest(train_dir, train=True, transform=train_transform, aug=aug, views=views, unique_patients=unique_patients)
            val_dataset = padchest.PadChest(train_dir, train=False, transform=val_transform, views=views, unique_patients=unique_patients)
    elif dataset_type == "merge":
        types = config.data.types.split(",")
        trains = []
        vals = []
        for t in types:
            cfg = deepcopy(config)
            cfg.data.type = t
            cfg.data.train_dir = getattr(config.data.train_dir,t)
            cfg.data.val_dir = getattr(config.data.val_dir,t)
            train, val = build_dataset(cfg)
            trains.append(train)
            vals.append(val)
        labels = set(trains[0].pathologies)
        for train in trains[1:]:
            labels = labels & set(train.pathologies)
        print(len(labels))
        labels = sorted(list(labels))
        label_inds = []
        for train in trains:
            label_inds.append([list(train.pathologies).index(l) for l in labels])
        train_dataset = Merge(trains, label_inds=label_inds)
        val_dataset = Merge(vals, label_inds=label_inds)
    elif dataset_type == "chexpert_mimic":
        config = deepcopy(config)
        
        train_dir = deepcopy(config.data.train_dir)
        val_dir = deepcopy(config.data.val_dir)
        config.data.type = "chexpert"
        config.data.train_dir = train_dir.chexpert
        config.data.val_dir = val_dir.chexpert
        train_chexpert, val_chexpert = build_dataset(config)

        config.data.type = "mimic"
        config.data.train_dir = train_dir.mimic
        config.data.val_dir = val_dir.mimic
        train_mimic, val_mimic = build_dataset(config)
        train_dataset = Merge([train_mimic, train_chexpert])
        val_dataset = Merge([val_mimic, val_chexpert])
    else:
        raise ValueError(f"Cannot recoginize dataset name: {dataset_type}")
    return train_dataset, val_dataset

class Merge:

    def __init__(self, datasets, label_inds=None):
        if hasattr(datasets[0], "classes"):
            self.classes = datasets[0].classes
        if hasattr(datasets[0], "transform"):
            self.transform = datasets[0].transform
        self.datasets = datasets
        self.dataset_inds = []
        self.offsets = []
        for i, ds in enumerate(datasets):
            self.dataset_inds.extend([i]*len(ds))
            self.offsets.extend(list(range(len(ds))))
        self.label_inds = label_inds

    def __getitem__(self, idx):
        dataset_ind = self.dataset_inds[idx]
        dataset = self.datasets[dataset_ind]
        x, y = dataset[self.offsets[idx]]
        if self.label_inds:
            y = y[self.label_inds[dataset_ind]]
        return x, y

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

def build_transforms(config):
    name = config.data.augmentation.type
    rand_augment = config.data.augmentation.rand_augment
    image_size = config.data.image_size
        
    if config.data.mean:
        mean = config.data.mean
    else:
        mean = [0.485, 0.456, 0.406]
    if config.data.std:
        std = config.data.std
    else:
        std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean, std)
    if name == "random_resized_crop":
        eval_resize = config.data.augmentation.eval_resize
        if not eval_resize:
            eval_resize = 256
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize(eval_resize),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    elif name == "bit":
        from transfer_learning.finetuning import bit_hyperrule
        precrop, image_size = bit_hyperrule.get_resolution_from_dataset(config.data.type)
        # precrop = config.data.precrop
        # if not precrop:
            # precrop = 256
        train_transform = transforms.Compose([
            transforms.Resize(precrop),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])
    elif name == "chexpert":
        import torchxrayvision as xrv
        size = len(mean)
        m = np.array(mean).reshape((size,1, 1))
        s = np.array(std).reshape((size, 1, 1))
        transform = torchvision.transforms.Compose([
            xrv.datasets.XRayCenterCrop(),
            xrv.datasets.XRayResizer(224, engine='cv2'),
            Normalize(m, s),
        ])
        train_transform = transform
        val_transform = transform
    else:
        raise ValueError(f"Cannot recognize transform: {name}")
    if rand_augment is not None:
        from RandAugment import RandAugment
        N = rand_augment.N if rand_augment.N else 2
        M = rand_augment.M if rand_augment.M else 9
        print(f"Using rand augment with ({N},{M})")
        train_transform.transforms.insert(0, RandAugment(N, M))
    return train_transform, val_transform

class Normalize:

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        n = x.shape[0]
        return (x - self.mean[:n]) / self.std[:n]
