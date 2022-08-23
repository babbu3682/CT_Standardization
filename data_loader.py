from torch.utils import data
from torchvision import transforms as vision_transforms
# from torchvision.datasets import ImageFolder
# from PIL import Image
import torch
import os
import random

from monai.transforms import *
from monai.data import Dataset
import pandas as pd


def Spacingd_Individual(x):
    y_spacing = abs(x['image_meta_dict']['affine'][1][1])
    x         = Spacingd(keys=["image"], pixdim=(y_spacing, y_spacing, 3.0), mode=('bilinear'))(x)

    return x
        
def CT_12bit_processing(x):
    # -1024 ~ 3071
    #     0 ~ 4095 : 4096 (2^12)

    x[x < -1024.0] = -1024.0
    x[x > 3071.0]  = 3071.0
    
    x = (x + 1024.0) / 4095.0
    
    # output : 0.0 ~ 1.0
    return x


# Read dataset
E_df = pd.read_excel('/workspace/sunggu/5.CT_Standardization/dataset/NII_Pair_CT_dataset/Enhance_O_df.xlsx', engine='openpyxl', index_col=0) 
# N_df = pd.read_excel('/workspace/sunggu/5.CT_Standardization/dataset/NII_Pair_CT_dataset/Enhance_X_df.xlsx', engine='openpyxl', index_col=0) 
Simens_label    = {'B50f':0,     'B30f':1,   'B70f':2}
GE_label        = {'STANDARD':0, 'DETAIL':1, 'CHST':2}
Philips_label   = {'YA':0,       'YC':1}
TOSHIBA_label   = {'FC08':0,     'FC04':1}

# Z, Spacing 고려가 안되어 있어,,,,

train_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),

        # interploation
        Spacingd_Individual,

        # Crop
        # RandWeightedCropd(keys=["image"], w_key=["image"], spatial_size=(512,512,1), num_samples=1),
        # RandSpatialCropd(keys=["image"], roi_size=(512,512,1), random_size=False, random_center=True),
        RandSpatialCropSamplesd(keys=["image"], roi_size=(64, 64, 1), num_samples=8, random_center=True, random_size=False),
        
        
        # Align
        Flipd(keys=["image"], spatial_axis=1),
        Rotate90d(keys=["image"], k=1, spatial_axes=(0, 1)),                

        # 12 비트 CT shifting 전처리
        Lambdad(keys=["image"], func=CT_12bit_processing), 

        ToTensord(keys=["image"]),
        Lambdad(keys=["image"], func=vision_transforms.Normalize(mean=(0.5), std=(0.5))),
    ]
)

test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        
        # interploation
        Spacingd_Individual,

        # Crop
        RandSpatialCropd(keys=["image"], roi_size=(512,512,1), random_size=False, random_center=True),

        # Align
        Flipd(keys=["image"], spatial_axis=1),
        Rotate90d(keys=["image"], k=1, spatial_axes=(0, 1)),

        # 12 비트 CT shifting 전처리
        Lambdad(keys=["image"], func=CT_12bit_processing), 

        ToTensord(keys=["image"]),
        Lambdad(keys=["image"], func=vision_transforms.Normalize(mean=(0.5), std=(0.5))),
    ]
)


### 1. Simens
def Simens_Dataset(mode):
    df        = E_df[(E_df['Convolution Kernel']=='B50f') | (E_df['Convolution Kernel']=='B30f') | (E_df['Convolution Kernel']=='B70f')]

    if mode == 'train':
        images    = df[df['Mode'] == 'Train']['Path'].values
        labels    = df[df['Mode'] == 'Train']['Convolution Kernel'].values
        ids       = df[df['Mode'] == 'Train']['Anonymized ID'].values

        files = [{"image":i, "label":Simens_label[l], 'id':id} for i, l, id in zip(images, labels, ids)]
        print("Train [Total]  number = ", len(images))

        return Dataset(data=files, transform=train_transforms)
 
    else :
        images    = df[df['Mode'] == 'Test']['Path'].values
        labels    = df[df['Mode'] == 'Test']['Convolution Kernel'].values
        ids       = df[df['Mode'] == 'Test']['Anonymized ID'].values

        files = [{"image":i, "label":Simens_label[l], 'id':id} for i, l, id in zip(images, labels, ids)]
        print("TEST [Total]  number = ", len(images))

        return Dataset(data=files, transform=test_transforms)

### 2. GE
def GE_Dataset(mode):
    df        = E_df[(E_df['Convolution Kernel']=='STANDARD') | (E_df['Convolution Kernel']=='DETAIL') | (E_df['Convolution Kernel']=='CHST')]

    if mode == 'train':
        images    = df[df['Mode'] == 'Train']['Path'].values
        labels    = df[df['Mode'] == 'Train']['Convolution Kernel'].values
        ids       = df[df['Mode'] == 'Train']['Anonymized ID'].values

        files = [{"image":i, "label":GE_label[l], 'id':id} for i, l, id in zip(images, labels, ids)]
        print("Train [Total]  number = ", len(images))

        return Dataset(data=files, transform=train_transforms)
 
    else :
        images    = df[df['Mode'] == 'Test']['Path'].values
        labels    = df[df['Mode'] == 'Test']['Convolution Kernel'].values
        ids       = df[df['Mode'] == 'Test']['Anonymized ID'].values

        files = [{"image":i, "label":GE_label[l], 'id':id} for i, l, id in zip(images, labels, ids)]
        print("TEST [Total]  number = ", len(images))

        return Dataset(data=files, transform=test_transforms)

### 3. Philips
def Philips_Dataset(mode):
    df        = E_df[(E_df['Convolution Kernel']=='YA') | (E_df['Convolution Kernel']=='YC')]
    if mode == 'train':    
        images    = df[df['Mode'] == 'Train']['Path'].values
        labels    = df[df['Mode'] == 'Train']['Convolution Kernel'].values
        ids       = df[df['Mode'] == 'Train']['Anonymized ID'].values

        files = [{"image":i, "label":Philips_label[l], 'id':id} for i, l, id in zip(images, labels, ids)]
        print("Train [Total]  number = ", len(images))

        return Dataset(data=files, transform=train_transforms)
 
    else :
        images    = df[df['Mode'] == 'Test']['Path'].values
        labels    = df[df['Mode'] == 'Test']['Convolution Kernel'].values
        ids       = df[df['Mode'] == 'Test']['Anonymized ID'].values

        files = [{"image":i, "label":Philips_label[l], 'id':id} for i, l, id in zip(images, labels, ids)]
        print("TEST [Total]  number = ", len(images))

        return Dataset(data=files, transform=test_transforms)

### 4. TOSHIBA
def TOSHIBA_Dataset(mode):
    df        = E_df[(E_df['Convolution Kernel']=='FC08') | (E_df['Convolution Kernel']=='FC04')]

    if mode == 'train':    
        images    = df[df['Mode'] == 'Train']['Path'].values
        labels    = df[df['Mode'] == 'Train']['Convolution Kernel'].values
        ids       = df[df['Mode'] == 'Train']['Anonymized ID'].values

        files = [{"image":i, "label":TOSHIBA_label[l], 'id':id} for i, l, id in zip(images, labels, ids)]
        print("Train [Total]  number = ", len(images))

        return Dataset(data=files, transform=train_transforms)
 
    else :
        images    = df[df['Mode'] == 'Test']['Path'].values
        labels    = df[df['Mode'] == 'Test']['Convolution Kernel'].values
        ids       = df[df['Mode'] == 'Test']['Anonymized ID'].values

        files = [{"image":i, "label":TOSHIBA_label[l], 'id':id} for i, l, id in zip(images, labels, ids)]
        print("TEST [Total]  number = ", len(images))

        return Dataset(data=files, transform=test_transforms)


def get_loader(batch_size=16, dataset='Simens', mode='train', num_workers=1):
    """Build and return a data loader."""

    if dataset == 'Simens':
        dataset = Simens_Dataset(mode)
    elif dataset == 'GE':
        dataset = GE_Dataset(mode)
    elif dataset == 'Philips':
        dataset = Philips_Dataset(mode)
    elif dataset == 'TOSHIBA':
        dataset = TOSHIBA_Dataset(mode)                


    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    return data_loader
