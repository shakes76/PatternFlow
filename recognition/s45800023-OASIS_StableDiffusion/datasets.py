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

train = modules.OASIS_Loader()
test = modules.OASIS_Loader(root_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/keras_png_slices_test/')
val = modules.OASIS_Loader(root_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/keras_png_slices_validation/')

# Initialize data loaders 

train_loader = DataLoader(train, batch_size=128, shuffle=True, drop_last=True, pin_memory=True)
test_loader = DataLoader(test, batch_size=128, shuffle=False, drop_last=False)
val_loader = DataLoader(val, batch_size=128, shuffle=False, drop_last=False)






































