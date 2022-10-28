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
import cv2
import datasets
import modules
from scipy.signal import savgol_filter

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

data = datasets.OASISData()
train, test, val = data.get_loaders()
hyperparameters = modules.Hyperparameters()
epochs = 5
VQVAE_PATH = "D:/Jacob Barrie/Documents/COMP3710/models/vqvaeNewBest.txt"


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

VQVAEtrain = torch.load(VQVAE_PATH) # load trained VQVAE
### Train DCGAN ###

# Initialise data
dataGan = datasets.DCGANLoader(VQVAEtrain)
train_gan, test_gan, val_gan = dataGan.get_loaders()
d_loss = []
g_loss = []

# Initalise models
Discriminator = modules.Discriminator(hyperparameters.channels_image,
                                      hyperparameters.features_d)
Generator = modules.Generator(hyperparameters.channels_noise,
                              hyperparameters.channels_image,
                              hyperparameters.features_g)

# Initialise combined
DCGAN = modules.trainDCGAN(Discriminator, Generator, train_gan)
DCGAN.train(d_loss, g_loss)