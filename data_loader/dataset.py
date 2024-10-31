from torch.utils.data import Dataset
from PIL import Image 
import torch 
import numpy as np 
import os 
from torchvision.transforms import transforms
from monai import transforms as transforms_monai
import albumentations as A
from glob import glob
import numpy as np
import pandas as pd 
import json 
from datetime import datetime
from copy import deepcopy
import pickle as pkl
import cv2
import SimpleITK as sitk

    

class PeritoneumDataset_Clinic(Dataset):
    def __init__(self, data_dir, split_file, split, N_lesion=5, target_size=(16, 144, 144), strong_aug=True, treatment_ids=[0,], thres=365, printer = print):
        super().__init__()
        self.data_dir = data_dir
        self.data_list = pd.read_csv(split_file)
        self.data_list = self.data_list[self.data_list["Treatment"].isin(treatment_ids)].values.tolist() 
        self.split = split
        self.label = self._build_label(self.data_list, thres=thres)
        self.target_size = target_size
        self.N_lesion = N_lesion
        self.printer = printer
        self.diag_labels = deepcopy(self.label)
        self.n_classes = len(np.unique(self.label))

        if strong_aug: #训练集
            self.transforms = transforms_monai.Compose(
                                    [
                                    transforms_monai.LoadImaged(keys=["image", "label"]),
                                    transforms_monai.EnsureChannelFirstd(keys=["image", "label"]),
                                    transforms_monai.Orientationd(keys=["image", "label"], axcodes="RAS"),
                                    transforms_monai.Spacingd(
                                        keys=["image", "label"], pixdim=(0.713, 0.713, 5.0), mode=("bilinear", "nearest")
                                    ),
                                    transforms_monai.ScaleIntensityRanged(
                                        keys=["image"], a_min=-125, a_max=225, b_min=0, b_max=1, clip=True
                                    ),
                                    transforms_monai.CropForegroundd(keys=["image", "label"], source_key="image"),
                                    transforms_monai.SpatialPadd(keys=["image", "label"], spatial_size=(144, 144, 16), mode='constant'),
                                    transforms_monai.CenterSpatialCropd(keys=["image", "label"], roi_size=(144, 144, 16)),
                                    transforms_monai.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=0),
                                    transforms_monai.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=1),
                                    transforms_monai.RandFlipd(keys=["image", "label"], prob=0.2, spatial_axis=2),
                                    transforms_monai.RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
                                    transforms_monai.RandScaleIntensityd(keys="image", factors=0.1, prob=0.1),
                                    transforms_monai.RandShiftIntensityd(keys="image", offsets=0.1, prob=0.1),
                                    transforms_monai.RandGaussianNoiseD(keys=["image"], mean=0.0, std=0.01, prob=0.10),
                                    transforms_monai.ToTensord(keys=["image", "label"]),
                                    ])
        else: #验证集测试集
            self.transforms = transforms_monai.Compose(
                                    [
                                        transforms_monai.LoadImaged(keys=["image", "label"]),
                                        transforms_monai.EnsureChannelFirstd(keys=["image", "label"]),
                                        transforms_monai.Orientationd(keys=["image", "label"], axcodes="RAS"),
                                        transforms_monai.Spacingd(
                                            keys=["image", "label"], pixdim=(0.713, 0.713, 5.0), mode=("bilinear", "nearest")
                                        ),
                                        transforms_monai.ScaleIntensityRanged(
                                            keys=["image"], a_min=-125, a_max=225, b_min=0, b_max=1, clip=True
                                        ),
                                        transforms_monai.CropForegroundd(keys=["image", "label"], source_key="image"),
                                        transforms_monai.SpatialPadd(keys=["image", "label"],spatial_size=(144, 144, 16), mode='constant'),
                                        transforms_monai.CenterSpatialCropd(keys=["image", "label"], roi_size=(144, 144, 16)),
                                        transforms_monai.ToTensord(keys=["image", "label"]),
                                    ]
                                    )
            
        self._summary()
        
    def __len__(self):
        return len(self.data_list)
    
    def _summary(self):
        self.printer(f"[{self.split}]\tLoaded {self.__len__()} samples")
        for i in [-1, 0, 1]:
            num = np.sum(self.label==i)
            self.printer(f"[Label={i}]\t{num} ({100.0*num/self.__len__():.2f}%)")
            
    def _build_label(self, data, thres=365):
        label = []
        for _, OS, OSCensor, _, PCI in data:
            if OS <= thres:
                if OSCensor == 1: #确定死亡，即无删失
                    label.append(0) #0表示高风险
                else:
                    label.append(-1) #有删失
            else:
                label.append(1) #1表示低风险
        return np.asarray(label)
    
    def __getitem__(self, index):
        d = self.data_list[index]
        y = self.label[index]
        
        pid, OS, OSCensor, Treatment, PCI = d
        pid = str(pid)
        pid_data_dir = os.path.join(self.data_dir, pid)
        mask_path = os.path.join(pid_data_dir, 'mask.nii.gz')
        mask_files = sorted(glob(os.path.join(pid_data_dir, "mask_*_16_144_144.nii.gz")))
        if self.N_lesion == 5:
            mask_files = [item for item in mask_files if 'mask_1_16_144_144.nii.gz' not in item]
        elif self.N_lesion == 1:
            mask_files = [item for item in mask_files if 'mask_1_16_144_144.nii.gz' in item]
        
        X = torch.zeros((self.N_lesion, 1, self.target_size[0], self.target_size[1], self.target_size[2])).float()
        MASK = torch.zeros((self.N_lesion, 1, self.target_size[0], self.target_size[1], self.target_size[2])).float()
        M = torch.zeros((self.N_lesion)).float()
        K = ["None"] * self.N_lesion
        for i, mask_file in enumerate(mask_files):
            idx = mask_file.split('/')[-1].split('_')[1]
            img_file = mask_file.replace('mask_{}_16_144_144.nii.gz'.format(idx), 'image_{}_16_144_144.nii.gz'.format(idx))
            data = {'image': img_file, 'label': mask_file}
            results = self.transforms(data) 
            img, mask = results['image'], results['label']
            img = img.permute(0, 3, 2, 1)
            mask = mask.permute(0, 3, 2, 1)
            X[i] = img
            MASK[i] = mask
            M[i] = 1.0
            K[i] = mask_file
        
        #年龄，性别，Lauren，是否免疫治疗，是否靶向治疗
        pathology = pd.read_excel('../data/internal/pathology.xlsx')

        sex = pathology[pathology['影像号'] == int(pid)]['性别（男=1，女=0）'].item()
        if pd.isna(sex):
            sex_label = 0
        elif sex == 0:
            sex_label = 1
        elif sex == 1:
            sex_label = 2
        else:
            raise ValueError()
        
        age = pathology[pathology['影像号'] == int(pid)]['是否超过60岁'].item()
        if pd.isna(age):
            age_label = 0
        elif age == 0:
            age_label = 1
        elif age == 1:
            age_label = 2
        else:
            raise ValueError()
        
        Lauren = pathology[pathology['影像号'] == int(pid)]['Lauren分型（0=肠型，1=弥漫型，2=混合型）'].item()
        if pd.isna(Lauren):
            Lauren_label = 0
        elif Lauren == 0:
            Lauren_label = 1
        elif Lauren == 1:
            Lauren_label = 2
        elif Lauren == 2:
            Lauren_label = 3
        else:
            raise ValueError()
        
        immunotherapy = pathology[pathology['影像号'] == int(pid)]['患者是否接受免疫治疗（是=1，否=0）'].item()
        if pd.isna(immunotherapy):
            immunotherapy_label = 0
        elif immunotherapy == 0:
            immunotherapy_label = 1
        elif immunotherapy == 1:
            immunotherapy_label = 2
        else:
            raise ValueError()
        
        target = pathology[pathology['影像号'] == int(pid)]['是否接受靶向治疗'].item()
        if pd.isna(target):
            target_label = 0
        elif target == 0:
            target_label = 1
        elif target == 1:
            target_label = 2
        else:
            raise ValueError()

        clicinfo = torch.LongTensor([
            sex_label,
            age_label,
            Lauren_label,
            immunotherapy_label,
            target_label,
        ])

        return {
            "pid": pid,
            "X": X,
            "MASK": MASK,
            "M": M,
            "y": y,
            "OS": OS,
            "OSCensor": OSCensor,
            "Treatment": Treatment,
            "K": K,
            "sex": sex_label,
            "age": age_label,
            "Lauren": Lauren_label,
            "immunotherapy": immunotherapy_label,
            "target": target_label,
            "clicinfo": clicinfo,
        }
    

    