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

# Initialise model
VQVAE = torch.load("D:/Jacob Barrie/Documents/COMP3710/models/vqvaeNewBest.txt")

# Initalise data
data = datasets.OASISData()
train, test, val = data.get_loaders()

# Obtain reconstruction, embedded slices
VQVAEpredict = modules.VQVAEpredict(VQVAE, test)
VQVAEpredict.reconstruction()
VQVAEpredict.embedding_slice()