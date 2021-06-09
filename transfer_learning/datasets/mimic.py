import pandas as pd
import numpy as np
import os
import albumentations as A
import torch
from torch.utils.data import Dataset
import cv2

class MIMIC(Dataset):
    """
    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY, Mark RG, Horng S. 
    MIMIC-CXR: A large publicly available database of labeled chest radiographs. 
    arXiv preprint arXiv:1901.07042. 2019 Jan 21.
    
    https://arxiv.org/abs/1901.07042
    
    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """
    # def __init__(self, imgpath, csvpath,metacsvpath, views=["PA"], transform=None, data_aug=None,
                 # flat_dir=True, seed=0, unique_patients=True):
    def __init__(self, path, version="chexpert", split="train", aug=None, transform=None, views=["AP", "PA"], unique_patients=False):
        super().__init__()
        splits = pd.read_csv(os.path.join(path, "mimic-cxr-2.0.0-split.csv.gz"))
        imgpath = os.path.join(path, "files")
        metacsvpath = os.path.join(path, "mimic-cxr-2.0.0-metadata.csv.gz")
        csvpath = os.path.join(path, f"mimic-cxr-2.0.0-{version}.csv.gz")
        # np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Opacity",
                            "Lung Lesion",
                            "Edema",
                            "Consolidation",
                            "Pneumonia",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Pleural Other",
                            "Fracture",
                            "Support Devices"]
        
        self.pathologies = sorted(self.pathologies)
        
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = aug
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.metacsvpath = metacsvpath
        self.metacsv = pd.read_csv(self.metacsvpath)
        
        self.csv = self.csv.set_index(['subject_id', 'study_id'])
        self.metacsv = self.metacsv.set_index(['subject_id', 'study_id'])
        
        self.csv = self.csv.join(self.metacsv).reset_index()

        # Keep only the desired view
        self.views = views
        if self.views:
            if type(views) is not list:
                views = [views]
            self.views = views

            self.csv["view"] = self.csv["ViewPosition"]
            self.csv = self.csv[self.csv["view"].isin(self.views)]

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()
                   
        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        self.labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
                
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        # make all the -1 values into nans to keep things simple
        self.labels[self.labels == -1] = np.nan
        
        # rename pathologies
        self.pathologies = np.char.replace(self.pathologies, "Pleural Effusion", "Effusion")
        
        ########## add consistent csv values
        
        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]
        
        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)
        
        df = self.csv.copy()
        df["ind"] = np.arange(len(df))
        df = pd.merge(df, splits, on=("dicom_id", "study_id", "subject_id"), how="left")
        df = df[df.split==split]
        self.csv = df
        self.labels = self.labels[df.ind.values]
        
    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])
        
        img_path = os.path.join(self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        img = cv2.imread(img_path)
        img = img / 255.0
        # img = normalize(img, self.MAXVAL)      
        
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            img = img[:, :, 0]
        if len(img.shape) < 2:
            print("error, dimension lower than 2 for image")

        # Add color channel
        img = img[None, :, :]

        if self.transform is not None:
            img = self.transform(img)
            
        if self.data_aug is not None:
            img = self.data_aug(img)

        img = img * np.ones((3,1,1), dtype="float32") # use 3 channels
        img = torch.from_numpy(img).float()
        target = torch.from_numpy(self.labels[idx]).float()
        return img, target 
        # return {"img":img, "lab":self.labels[idx], "idx":idx}
