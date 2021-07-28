import pandas as pd
import numpy as np
import os
import albumentations as A
import torch
from torch.utils.data import Dataset
import random
# import cv2
from skimage.io import imread
SEED = 42
TRAIN_RATIO = 0.9

class PadChest(Dataset):
    """
    PadChest dataset
    Hospital San Juan de Alicante - University of Alicante
    
    PadChest: A large chest x-ray image dataset with multi-label annotated reports.
    Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria de la Iglesia-Vayá. 
    arXiv preprint, 2019. https://arxiv.org/abs/1901.07441
    
    Dataset website:
    http://bimcv.cipf.es/bimcv-projects/padchest/
    
    Download full size images here:
    https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850
    
    Download resized (224x224) images here (recropped):
    https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797
    """
    def __init__(self, path, train=True, aug=None, transform=None, views=["AP", "PA"], unique_patients=False):
    # def __init__(self, imgpath, 
                 # csvpath=os.path.join(thispath, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"), 
                 # views=["PA"],
                 # transform=None, 
                 # data_aug=None,
                 # flat_dir=True, 
                 # seed=0, 
                 # unique_patients=True):
        super().__init__()
        # super(PC_Dataset, self).__init__()
        # np.random.seed(seed)  # Reset the seed so all runs are the same.
        csvpath = os.path.join(path, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz")
        data_aug = aug

        self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia","Fracture", 
                            "Granuloma", "Flattened Diaphragm", "Bronchiectasis",
                            "Aortic Elongation", "Scoliosis", 
                            "Hilar Enlargement", "Support Devices" , "Tuberculosis",
                            "Air Trapping", "Costophrenic Angle Blunting", "Aortic Atheromatosis",
                            "Hemidiaphragm Elevation"]
        
        self.pathologies = sorted(self.pathologies)
        
        mapping = dict()
        
        mapping["Infiltration"] = ["infiltrates",
                                   "interstitial pattern", 
                                   "ground glass pattern",
                                   "reticular interstitial pattern",
                                   "reticulonodular interstitial pattern",
                                   "alveolar pattern",
                                   "consolidation",
                                   "air bronchogram"]
        mapping["Pleural_Thickening"] = ["pleural thickening"]
        mapping["Consolidation"] = ["air bronchogram"]
        mapping["Hilar Enlargement"] = ["adenopathy",
                                        "pulmonary artery enlargement"]
        mapping["Support Devices"] = ["device",
                                      "pacemaker"]
        
        self.imgpath = path
        self.transform = transform
        self.data_aug = data_aug
        # self.flat_dir = flat_dir
        self.csvpath = csvpath
        
        # self.check_paths_exist()
        self.csv = pd.read_csv(self.csvpath, low_memory=False)
        # self.MAXVAL = 65535

        # standardize view names
        self.csv.loc[self.csv["Projection"].isin(["AP_horizontal"]),"Projection"] = "AP Supine"
        
        # Keep only the specified views
        if type(views) is not list:
            views = [views]
        self.views = views
        
        self.csv["view"] = self.csv['Projection']
        # print(self.csv.view.unique())
        self.csv = self.csv[self.csv["view"].isin(self.views)]

        # remove null stuff
        self.csv = self.csv[~self.csv["Labels"].isnull()]
        
        # remove missing files
        missing = ["216840111366964012819207061112010307142602253_04-014-084.png",
                   "216840111366964012989926673512011074122523403_00-163-058.png",
                   "216840111366964012959786098432011033083840143_00-176-115.png",
                   "216840111366964012558082906712009327122220177_00-102-064.png",
                   "216840111366964012339356563862009072111404053_00-043-192.png",
                   "216840111366964013076187734852011291090445391_00-196-188.png",
                   "216840111366964012373310883942009117084022290_00-064-025.png",
                   "216840111366964012283393834152009033102258826_00-059-087.png",
                   "216840111366964012373310883942009170084120009_00-097-074.png",
                   "216840111366964012819207061112010315104455352_04-024-184.png"]
        missing.extend([
	    # "216840111366964012283393834152009033102258826_00-059-087.png",
	    # "216840111366964012339356563862009068084200743_00-045-105.png",
	    # "216840111366964012339356563862009072111404053_00-043-192.png",
	    # "216840111366964012373310883942009117084022290_00-064-025.png",
	    # "216840111366964012373310883942009170084120009_00-097-074.png",
	    # "216840111366964012558082906712009300162151055_00-078-079.png",
	    # "216840111366964012558082906712009327122220177_00-102-064.png",
	    # "216840111366964012819207061112010306085429121_04-020-102.png",
	    # "216840111366964012819207061112010307142602253_04-014-084.png",
	    # "216840111366964012819207061112010315104455352_04-024-184.png",
	    # "216840111366964012959786098432011033083840143_00-176-115.png",
	    # "216840111366964012989926673512011074122523403_00-163-058.png",
	    # "216840111366964012989926673512011101154138555_00-191-086.png",
	    # "216840111366964012989926673512011132200139442_00-157-099.png",
	    # "216840111366964013076187734852011178154626671_00-145-086.png",
	    # "216840111366964013076187734852011291090445391_00-196-188.png",
            #wrong
            "216840111366964013829543166512013353113303615_02-092-190.png",
            "216840111366964012904401302362010337093236130_03-198-079.png",
            "216840111366964012904401302362010336141343749_03-198-010.png",
            "216840111366964012989926673512011151082430686_00-157-045.png",
            "216840111366964012989926673512011083134050913_00-168-009.png",
            "216840111366964012373310883942009077082646386_00-047-124.png",
            "216840111366964013686042548532013208193054515_02-026-007.png",
            "216840111366964013962490064942014134093945580_01-178-104.png",
            "216840111366964012819207061112010281134410801_00-129-131.png",
            "216840111366964013590140476722013043111952381_02-065-198.png",
            "216840111366964012283393834152009027091819347_00-007-136.png",
            "216840111366964012373310883942009152114636712_00-102-045.png",
            "216840111366964012283393834152009033140208626_00-059-118.png",
            "216840111366964013590140476722013058110301622_02-056-111.png",
            "216840111366964012487858717522009280135853083_00-075-001.png",
            "216840111366964013590140476722013049100117076_02-063-097.png",
            "216840111366964013649110343042013092101343018_02-075-146.png",
            "216840111366964012487858717522009280135853083_00-075-001.png",
            "216840111366964012819207061112010306085429121_04-020-102.png",
            "269300710246070740096540277379121868595_e7zsan.png",
            "216840111366964012373310883942009180082307973_00-097-011.png",
        ])
        self.csv = self.csv[~self.csv["ImageID"].isin(missing)]
        
        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()
        
        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv["Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    #print("mapping", syn)
                    mask |= self.csv["Labels"].str.contains(syn.lower())
            self.labels.append(mask.values)
        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        
        ########## add consistent csv values
        
        # offset_day_int
        dt = pd.to_datetime(self.csv["StudyDate_DICOM"], format="%Y%m%d")
        self.csv["offset_day_int"] = dt.astype(np.int)// 10**9 // 86400
        
        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)


        inds = np.arange(len(self.csv))
        rng = np.random.RandomState(SEED)
        rng.shuffle(inds)
        # print("Padchest size full" , len(self.csv))
        nb_train = int(len(inds) * TRAIN_RATIO)
        if train:
            inds = inds[0:nb_train]
        else:
            inds = inds[nb_train:]
        self.csv = self.csv.iloc[inds]
        self.labels = self.labels[inds]
        # print("Padchest size" , len(self.csv))

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        imgid = self.csv['ImageID'].iloc[idx]
        img_path = os.path.join(self.imgpath,imgid)
        # try:
        img = imread(img_path)
        # except Exception:
            # print('<<',img_path,'>>')
            # return torch.zeros((3,224,224)).float(),torch.zeros(27).float()
        img = img / 65535
        # print(img.min(), img.max())
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
