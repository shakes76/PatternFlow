# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:11:40 2022

Script for pre-processing and cleaning of OASIS brain dataset. 
Will output a dataloader to be used in conjunction with the training
script. 

@author: Jacob Barrie: s45800023
"""

import torch
import torchvision
from torchvision import transforms 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage import color, io, transform
import os
import cv2
import modules

device = modules.check_cuda()

# Initialize our training, test and validation sets. 

class OASISData():
    def __init__(self, img_dir='D:/Jacob Barrie/Documents/keras_png_slices_data'):
        data_transforms = [
            transforms.ToTensor(),
        ]  
        data_transform = transforms.Compose(data_transforms)
        self.train = OASIS_Loader(root_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/train/', transform=data_transform)
        self.test = OASIS_Loader(root_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/test/', transform=data_transform)
        self.val = OASIS_Loader(root_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/val/', transform=data_transform)
        
    # Initialize data loaders 
    def get_loaders(self):
        train_loader = DataLoader(self.train, batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
        test_loader = DataLoader(self.test, batch_size=32, shuffle=False, drop_last=False)
        val_loader = DataLoader(self.val, batch_size=32, shuffle=False, drop_last=False)
        return train_loader, test_loader, val_loader


class OASIS_Loader(Dataset):
    """
    Custom Dataset class for the OASIS dataset. 
    """
    
    def __init__(self, root_dir='D:/Jacob Barrie/Documents/keras_png_slices_data/',
                 transform = None):
        """
        Paramaters
        ----------
            root_dir (string): Path to directory containing images. 
            transform (callable, optional): Optional transform to be applied to data.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        
    def __len__(self):
        return int(len([name for name in os.listdir(self.root_dir)]))
    
    def __getitem__(self, idx):
        """
        Custom getitem method to ensure image files can obtained correctly. 
        """
        img_names = modules.get_filenames(self.root_dir) 
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, img_names[idx]) # Finds file path based on index
        image = cv2.imread(img_name) # Reads image
        sample = image    
        
        if self.transform: # Will apply image transform if required. 
            sample = self.transform(sample)    
            
        return sample



































