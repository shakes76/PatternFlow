# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:11:40 2022

Source code to train, test, validate and save the developed model. 

@author: Jacob Barrie: s45800023
"""

import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import cv2
import datasets
import modules
from scipy.signal import savgol_filter

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

"""
Sanity checking CUDA and setting the device. 

"""
data = datasets.OASISData()
train, test, val = data.get_loaders()
hyperparameters = modules.Hyperparameters()
epochs = 5

### Train VQVAE ###
VQVAE = modules.VQVAEtrain(hyperparameters, train, data) 
train_res_recon_error = []

for i in range(epochs):
    print("Epoch: ", i)
    VQVAE.train(train_res_recon_error)
    
# Plot loss
train_res_recon_error_smooth = savgol_filter(train_res_recon_error, 201, 7)
f = plt.figure(figsize=(16,8))
ax = f.add_subplot(1,2,1)
ax.plot(train_res_recon_error_smooth)
ax.set_yscale('log')
ax.set_title('Loss over 5 Epochs.')
ax.set_xlabel('iteration')

### Train DCGAN ###
