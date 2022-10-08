# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:29:31 2022

@author: blobf
"""
import torch
import torch.nn as nn
import torch.utils as utils
import torchvision
import numpy as np

import modules
import dataset

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


device = "cuda" if torch.cuda.is_available() else "cpu"


class Visualise():
    
    def __init__(self, model, data):
        #load the trained model from a pt file
        self.model = model
        
        #load the image data
        
        self.dataset = data
        
    def visualise_VQVAE(self):
        # generate an image every epoch
        real_img = self.data[0][0]
        _, ax = plt.subplots(1,2)
        
        real_img = real_img.view(-1, 3, 256,256).to(self.device).detach()
        real_grid = torchvision.utils.make_grid(real_img, normalize = True)
        decoded_img , _ = self.model(real_img)
        decoded_img = decoded_img.view(-1, 3, 256,256).to(self.device).detach()
        decoded_grid = \
            torchvision.utils.make_grid(decoded_img, normalize = True)
        decoded_grid = decoded_grid.to("cpu").permute(1,2,0)
        real_grid = real_grid.to("cpu").permute(1,2,0)
        ax[1].imshow(decoded_grid)
        ax[0].imshow(real_grid)
        plt.show()
        print(ssim(real_grid.numpy(), decoded_grid.numpy(), channel_axis = -1))
        