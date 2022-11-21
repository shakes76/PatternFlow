# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:11:40 2022

Script to run 'predictions' - in this case produce generated images of the
OASIS brain dataset.

@author: Jacob Barrie: s45800023
"""

import datasets
import modules
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

## VQVAE ##
VQVAE_PATH = "D:/Jacob Barrie/Documents/COMP3710/models/vqvaeNewBest.txt"
GENERATOR_PATH = "D:/Jacob Barrie/Documents/COMP3710/models/generator.txt"
train_dir = 'D:/Jacob Barrie/Documents/keras_png_slices_data/train/'

# Initialise model
VQVAE = torch.load(VQVAE_PATH)

# Initalise data
data = datasets.OASISData()
train, test, val = data.get_loaders()


# Obtain reconstruction, embedded slices
VQVAEpredict = modules.VQVAEpredict(VQVAE, test)
VQVAEpredict.reconstruction()
VQVAEpredict.embedding_slice()

## DCGAN ##
Generator = torch.load(GENERATOR_PATH)
generate = modules.generateDCGAN(Generator, VQVAE) 
generated = generate.gen()  # Generate fake codebook indice
decoded = generate.reconstruct(generated) # Decode and display
ssim = generate.SSIM(decoded, train_dir, train)