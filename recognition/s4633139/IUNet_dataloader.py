#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#  Copyright (c) 2021, H.WAKAYAMA, All rights reserved.
#  File: IUNet_dataloader.py
#  Author: Hideki WAKAYAMA
#  Contact: h.wakayama@uq.net.au
#  Platform: macOS Big Sur Ver 11.2.1, Pycharm pro 2021.1
#  Time: 19/10/2021, 15:47
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
from torch.utils.data import Dataset
from PIL import Image

os.chdir("./ISIC2018_Task1-2_Training_Data")

class UNet_dataset(Dataset):
    def __init__(self,
                 img_dir='./ISIC2018_Task1-2_Training_Input_x2',
                 mask_dir='./ISIC2018_Task1_Training_GroundTruth_x2',
                 img_transforms=None,
                 mask_transforms=None,
                 ):

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transforms = img_transforms
        self.mask_transforms = mask_transforms
        self.imgs = [file for file in sorted(os.listdir(self.img_dir)) if file.endswith('.jpg')]
        self.masks = [file for file in sorted(os.listdir(self.mask_dir)) if file.endswith('.png')]

    def load_data(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        return img, mask

    def __getitem__(self, idx):
        img, mask = self.load_data(idx)
        if self.img_transforms is not None:
            img = self.img_transforms(img)
        if self.mask_transforms is not None:
            mask = self.mask_transforms(mask)
        return img, mask

    def __len__(self):
        return len(self.imgs)