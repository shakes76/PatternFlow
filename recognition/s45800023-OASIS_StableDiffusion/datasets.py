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
    def __init__(self, img_dir='D:/Jacob Barrie/Documents/keras_png_slices_data/keras_png_slices_test/',):
        data_transforms = [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]  
        data_transform = transforms.Compose(data_transforms)
        self.train = modules.OASIS_Loader(transform=data_transform)
        self.test = modules.OASIS_Loader(root_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/keras_png_slices_test/', transform=data_transform)
        self.val = modules.OASIS_Loader(root_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/keras_png_slices_validation/', transform=data_transform)

    # Initialize data loaders 
    def get_loaders(self):
        train_loader = DataLoader(self.train, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
        test_loader = DataLoader(self.test, batch_size=128, shuffle=False, drop_last=False)
        val_loader = DataLoader(self.val, batch_size=128, shuffle=False, drop_last=False)
        return train_loader, test_loader, val_loader






































