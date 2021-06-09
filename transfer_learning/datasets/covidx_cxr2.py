import pandas as pd
import os
from glob import glob
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from numpy.random import RandomState

class COVIDX_CXR2(Dataset):

    def __init__(self, folder, train=True, transform=None):
        self.folder = folder
        self.transform = transform
        split = "train" if train else "test"
        df = pd.read_csv(os.path.join(folder, split+".txt"), names=['index', 'path', 'label', 'data_source'], sep=" ")
        self.files = [os.path.join(folder, split, path) for path in df.path.values]
        self.targets = [1 if label == "positive" else 0 for label in df.label.values]
        self.classes = [0, 1]

    def __getitem__(self, index):
        path = self.files[index]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        class_id = self.targets[index]
        return img, class_id

    def __len__(self):
        return len(self.files)
