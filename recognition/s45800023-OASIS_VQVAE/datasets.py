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
import os
import cv2
import modules

device = modules.check_cuda()
train_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/train/'
test_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/test/'
val_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/val/'
# Initialize our training, test and validation sets. 

class OASISData():
    def __init__(self, img_dir='D:/Jacob Barrie/Documents/keras_png_slices_data'):
        # Initialize tranforms
        data_transforms = [
            torchvision.transforms.ToTensor(),
        ]  
        data_transform = transforms.Compose(data_transforms)
        
        # Create datasets
        self.train = OASIS_Loader(root_dir = train_dir, transform=data_transform)
        self.test = OASIS_Loader(root_dir = test_dir, transform=data_transform)
        self.val = OASIS_Loader(root_dir = val_dir, transform=data_transform)
        
    # Initialize data loaders 
    def get_loaders(self):
        """
        Method to return data loaders.
        """
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


class dataDCGAN(Dataset):
    """
    We are training DCGAN on the encodings from the VQVAE in so we can then use
    the DCGAN to generate images in the latent space. This class creates a new
    encoded version of the OASIS dataset that can be used in training. 
    """
    def __init__(self, VQVAE, root_dir,
                 transform = None):
        """
        Paramaters
        ----------
            root_dir (string): Path to directory containing images. 
            transform (callable, optional): Optional transform to be applied to data.
        """
        self.device = torch.device("cuda")
        self.VQVAE = VQVAE
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
        sample = self.transform(sample)
        sample = sample.to(self.device)
        sample = sample.unsqueeze(0)
        
        # Here we follow the VQVAE model but we will not decode
        encoded = self.VQVAE.encoder(sample)
        conv = self.VQVAE.conv(encoded)
        _, _, _,embeddings = self.VQVAE.vq(conv)
        embeddings = embeddings.float().to(self.device)
        embeddings= embeddings.view(64,64)
        embeddings = torch.stack((embeddings, embeddings, embeddings), 0)

        return embeddings
    
class DCGANLoader():
    def __init__(self, VQVAE, img_dir='D:/Jacob Barrie/Documents/keras_png_slices_data'):
        # Initialise transforms
        data_transforms = [
            transforms.ToTensor()
        ]  
        data_transform = transforms.Compose(data_transforms)
        
        # Initialise trained VQVAE and create datasets
        self.VQVAE = VQVAE
        self.train = dataDCGAN(self.VQVAE, root_dir = train_dir, transform=data_transform)
        self.test = dataDCGAN(self.VQVAE, root_dir = test_dir, transform=data_transform)
        self.val = dataDCGAN(self.VQVAE, root_dir = val_dir, transform=data_transform)
        
    # Initialize data loaders 
    def get_loaders(self):
        """
        Method to return data loaders.
        """
        train_loader = DataLoader(self.train, batch_size=256, shuffle=True)
        test_loader = DataLoader(self.test, batch_size=32, shuffle=False)
        val_loader = DataLoader(self.val, batch_size=32, shuffle=False)
        return train_loader, test_loader, val_loader































