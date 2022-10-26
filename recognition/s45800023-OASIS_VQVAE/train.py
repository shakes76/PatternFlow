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
Sanity checking CUDA and setting the device. 

"""
device = modules.check_cuda()
