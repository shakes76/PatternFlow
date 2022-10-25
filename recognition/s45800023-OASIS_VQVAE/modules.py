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
        
class Residual(nn.Module):
    """
    Class defining a residual connection for the encoder/decoder classes.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, 
                      stride=1, 
                      bias=False)
        )
    
    def forward(self, x):
        return x + self.block(x)
    
class ResidualBlock(nn.Module):
    """
    Class defining a residual block to be used in the encoder/decoder classes.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualBlock, self).__init__()
        self.res1 = Residual(in_channels, num_hiddens, num_residual_hiddens)
        self.res2 = Residual(in_channels, num_hiddens, num_residual_hiddens)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        out = self.relu(out)
        return out
        
        
class Encoder(torch.nn.Module):
    """
    Encoder class for VAE.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self.relu2 = nn.ReLU()
        self.conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self.resBlock1 = ResidualBlock(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_hiddens=num_residual_hiddens)
        self.resBlock2 = ResidualBlock(in_channels=num_hiddens,
                                            num_hiddens=num_hiddens,
                                            num_residual_hiddens=num_residual_hiddens)
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.relu1(out)
        out = self.conv_2(out)
        out = self.relu2(out)
        out = self.conv_3(out)
        out = self.resBlock1(out)
        out = self.resBlock2(out)
        out = self.relu3(out)
        return out

    
        
class Decoder(torch.nn.Module):
    """
    Decoder class for VAE.
    """
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self.conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self.resBlock1 = ResidualBlock(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self.convT1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self.convT2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=3,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv_1(x)
        out = self.resBlock1(out)
        out = self.convT1(out)
        out = self.relu(out)
        out = self.convT2(out)
        return out

    

    


    