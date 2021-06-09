import pandas as pd
import numpy as np
import os
import albumentations as A
import torch
from torch.utils.data import Dataset
import random
import cv2

class NIH(Dataset):
    """
    NIH ChestX-ray8 dataset
    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community
    
    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a
    
    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """
    def __init__(self, path, train=True, aug=None, transform=None, views=["AP", "PA"], unique_patients=False):
    # def __init__(self, imgpath, 
                 # csvpath=os.path.join(thispath, "Data_Entry_2017_v2020.csv.gz"),
                 # bbox_list_path=os.path.join(thispath, "BBox_List_2017.csv.gz"),
                 # views=["PA"],
                 # transform=None, 
                 # data_aug=None, 
                 # nrows=None, 
                 # seed=0,
                 # pure_labels=False, 
                 # unique_patients=True,
                 # normalize=True,
                 # pathology_masks=False):
        
        super().__init__()
        pathology_masks = False
        pure_labels = False
        nrows = None

        imgpath = os.path.join(path, "images")
        csvpath = os.path.join(path, f"Data_Entry_2017_v2020.csv")
        bbox_list_path = os.path.join(path, "BBox_List_2017.csv")
        # np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = aug
        self.pathology_masks = pathology_masks
        
        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia"]
        
        self.pathologies = sorted(self.pathologies)
        
        # self.normalize = normalize
        # Load data
        # self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)
        # self.MAXVAL = 255  # Range [0 255]
        
        if type(views) is not list:
            views = [views]
        self.views = views
        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.csv = self.csv[self.csv["view"].isin(self.views)]
        
        # Remove multi-finding images.
        if pure_labels:
            self.csv = self.csv[~self.csv["Finding Labels"].str.contains("\|")]
        
        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()
        
        self.csv = self.csv.reset_index()
        
        ####### pathology masks ########
        # load nih pathology masks
        self.pathology_maskscsv = pd.read_csv(bbox_list_path, 
                names=["Image Index","Finding Label","x","y","w","h","_1","_2","_3"],
               skiprows=1)
        
        # change label name to match
        self.pathology_maskscsv["Finding Label"][self.pathology_maskscsv["Finding Label"] == "Infiltrate"] = "Infiltration"
        self.csv["has_masks"] = self.csv["Image Index"].isin(self.pathology_maskscsv["Image Index"])
        ####### pathology masks ########    
            
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            self.labels.append(self.csv["Finding Labels"].str.contains(pathology).values)
            
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        ########## add consistent csv values
        
        # offset_day_int
        #self.csv["offset_day_int"] = 
        
        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)
        
        df = self.csv.copy()
        if train:
            split = pd.read_csv(os.path.join(path, "train_val_list.txt"), names=["Image Index"])
        else:
            split = pd.read_csv(os.path.join(path, "test_list.txt"), names=["Image Index"])
        df["ind"] = np.arange(len(df))
        df = pd.merge(df, split, on=("Image Index",), how="inner")
        print(len(df), len(split))
        self.csv = df
        self.labels = self.labels[df.ind.values]

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        
        
        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        #print(img_path)
        img = cv2.imread(img_path)
        img = img / 255.0
        # if self.normalize:
            # img = normalize(img, self.MAXVAL)  

        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        sample["img"] = img[None, :, :]                    

        # transform_seed = np.random.randint(2147483647)
        
        if self.pathology_masks:
            sample["pathology_masks"] = self.get_mask_dict(imgid, sample["img"].shape[2])
            
        if self.transform is not None:
            # random.seed(transform_seed)
            sample["img"] = self.transform(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.transform(sample["pathology_masks"][i])
  
        if self.data_aug is not None:
            # random.seed(transform_seed)
            sample["img"] = self.data_aug(sample["img"])
            if self.pathology_masks:
                for i in sample["pathology_masks"].keys():
                    random.seed(transform_seed)
                    sample["pathology_masks"][i] = self.data_aug(sample["pathology_masks"][i])
            
        img = sample["img"]
        lab = sample["lab"]
        img = img * np.ones((3,1,1), dtype="float32") # use 3 channels
        img = torch.from_numpy(img).float()
        target = torch.from_numpy(lab).float()
        return img, target
