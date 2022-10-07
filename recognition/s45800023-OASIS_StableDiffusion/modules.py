# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:11:40 2022

Modules file: Contains helper functions/Classes to be used in conjunction
with main scripts. 

@author: Jacob Barrie: s45800023
"""

import torch
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
  
class Encoder(torch.nn.Module):
    """
    Encoder class for the diffusion model. Stable Diffusion makes use 
    of an auto-encoder to encode the image into the latent space 
    before performing diffusion.
    """
    def __init(self, input_channels=3):
        super().__init__()
        
class Decoder(torch.nn.Module):
    """
    Decoder class for after noise has been decoded by UNet. Transforms encoded
    denoised images in the latent space back into the pixel space. 
    """
    def __init__(self):
        super().__init__()
    
class UNet():
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


    