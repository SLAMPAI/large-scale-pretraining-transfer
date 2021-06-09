import os
from glob import glob
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from numpy.random import RandomState

SEED = 42
RATIO_TRAIN = 0.85

class Tuberculosis(Dataset):

    def __init__(self, folder, train=True, transform=None):
        self.folder = folder
        self.transform = transform
        rng = RandomState(SEED)
        files = sorted(glob(os.path.join(folder, "*.png")))
        rng.shuffle(files)
        nb_train = int(len(files) * RATIO_TRAIN)
        if train:
            self.files = files[0:nb_train]
        else:
            self.files = files[nb_train:]
        self.classes = [0, 1]

    def __getitem__(self, index):
        path = self.files[index]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        class_id = int(os.path.basename(path)[-5])#class is encoded in filename
        return img, class_id

    def __len__(self):
        return len(self.files)
