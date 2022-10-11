# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:11:40 2022

Modules file: Contains helper functions/Classes to be used in conjunction
with main scripts. 

@author: Jacob Barrie: s45800023
"""

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from skimage import color, io, transform
import os, os.path
import cv2

### HELPER FUNCTIONS ###

def get_filenames(root_dir):
    """
    Parameters
    ----------
    root_dir (string): Directory containing image files

    Returns
    -------
    List (strings): List containing file names.

    """
    return os.listdir(root_dir)

def plot_sample(data):
        """
        Method to plot a random sample of the images. Typically required only
        as a sanity check.

        """
        sample_idx = random.sample(range(1, len(data)), 3)
        fig = plt.figure()
        
        for i in range(len(sample_idx)):
            ax = plt.subplot(1, 3, i+1)
            plt.tight_layout()
            ax.set_title("Sample #{}".format(i+1))
            ax.axis('off')
            ax.imshow(data[sample_idx[i]])
        plt.show()
        
def check_cuda():
    """
    Method for sanity checking CUDA and setting the device. 

    """
    print("Is device available: ",torch.cuda.is_available())
    print("Current Device: ",torch.cuda.current_device())
    print("Name: ",torch.cuda.get_device_name(0))
    
    device = torch.cuda.device(0)
    return device 

#############################

### Modules ###
class OASIS_Loader(Dataset):
    """
    Custom Dataset class for the OASIS dataset. 
    """
    
    def __init__(self, root_dir='D:/Jacob Barrie/Documents/keras_png_slices_data/keras_png_slices_train/',
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
        img_names = get_filenames(self.root_dir) 
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, img_names[idx]) # Finds file path based on index
        image = cv2.imread(img_name) # Reads image
        sample = image    
        
        if self.transform: # Will apply image transform if required. 
            sample = self.transform(sample)    
            
        return sample
    
class Hyperparameters():
    """
    Class to initialize hyperparameters for use in training
    """
    def __init__(self):
        self.epochs = 10
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam()
        self.batch_size = 128
        self.lr = 0.99
        
        
  
class Encoder(torch.nn.Module):
    """
    Encoder class for the diffusion model. Stable Diffusion makes use 
    of an auto-encoder to encode the image into the latent space 
    before performing diffusion.
    """
    def __init(self, 
               input_channels=1,
               base_channel_size=32,
               latent_dim = 100,
               activation : object = nn.LeakyReLU()):
        super().__init__()
        c_d = base_channel_size
        self.net = nn.Sequential(
            # Block 1
            nn.Conv2d(input_channels, c_d, kernal_size=3, padding=1, stride=2),
            nn.BatchNorm2d(128*128*32),
            activation(),
            
            # Block 2
            nn.Conv2d(c_d,2*c_d, kernal_size=3, padding=1, stride=2),
            nn.BatchNorm2d(64*64*64),
            activation(),
            
            # Block 3
            nn.Conv2d(2*c_d,2*c_d, kernal_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32*32*64),
            activation(),
            
            # Block 4
            nn.Conv2d(2*c_d,2*c_d, kernal_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16*16*64),
            activation(),
            
            # Block 5
            nn.Conv2d(2*c_d,2*c_d, kernal_size=3, padding=1, stride=2),
            nn.BatchNorm2d(8*8*64),
            activation(),
            
            # Bottleneck
            nn.Flatten(),
            nn.Linear(8*8*64, 100)
            )
    def forward(self, x):
        return self.net(x)
    
        
class Decoder(torch.nn.Module):
    """
    Decoder class for after noise has been decoded by UNet. Transforms encoded
    denoised images in the latent space back into the pixel space. 
    """
    def __init__(self):
        super().__init__()
    
class UNet(torch.nn.Module):
    """
    U-Net class for diffusion model. The U-Net takes noisy encoded images 
    (noise introduced by diffusion model) and reconstructs the image. 
    """
    def __init__(self):
        super().__init__()
    
class Diffusion(torch.nn.Module):
    def __init(self):
        super().__init__()

def train():
    return


    