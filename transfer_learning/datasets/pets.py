import numpy as np
import pandas as pd
import os
from glob import glob
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

class Pets(Dataset):

    def __init__(self, folder, train=True, transform=None):
        self.folder = folder
        self.transform = transform
        split = "trainval.txt" if train else "test.txt"
        data = pd.read_csv(os.path.join(folder, "annotations", split), names=["file", "unk0", "unk1", "unk2"], sep=" ")
        labels = data.file.apply(get_class_name)
        self.paths = [os.path.join(folder, "images", f+".jpg") for f in data.file.values]
        self.classes = sorted(labels.unique().tolist())
        self.targets = [self.classes.index(l) for l in labels]

    def __getitem__(self, index):
        path = self.paths[index]
        class_id = self.targets[index]
        img = default_loader(path)
        if self.transform:
            img = self.transform(img)
        return img, class_id

    def __len__(self):
        return len(self.paths)

def get_class_name(filename):
    return "_".join(filename.split("_")[0:-1])
