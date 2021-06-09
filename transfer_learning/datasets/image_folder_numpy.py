import torch
from torchvision.datasets import ImageFolder
import cv2

class ImageFolderNumpy(ImageFolder):

    def __getitem__(self, idx):
        path, class_ = self.imgs[idx]
        img = cv2.imread(path)
        img = img[:, :, ::-1]
        img = img / 255.0 
        if self.transform is not None:
            img = self.transform(img)
        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img)
        return img, class_
