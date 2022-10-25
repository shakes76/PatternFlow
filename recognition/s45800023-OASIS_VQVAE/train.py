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
import os
import cv2
import datasets
import modules

"""
Code Block for sanity checking CUDA and setting the device. 

"""
print("Is device available: ",torch.cuda.is_available())
print("Current Device: ",torch.cuda.current_device())
print("Name: ",torch.cuda.get_device_name(0))

device = torch.cuda.device(0)

train = datasets.train