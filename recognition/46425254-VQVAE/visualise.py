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


class VQVAE_Visualise():
    
    def __init__(self, VQVAE_path, num_embeddings, latent_dim, data):
        #load the trained model from a pt file
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = modules.VQVAE(num_embeddings, latent_dim, 0.25)
        state_dict = torch.load(VQVAE_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self.model = model
        
        
        #load the image data
        self.dataset = data
        
    def visualise_VQVAE(self, coords): 
        real_img = self.dataset[coords[0]][coords[1]]
        _, ax = plt.subplots(1,2)
        
        real_img = real_img.view(-1, 3, 128,128).to(device).detach()
        real_grid = torchvision.utils.make_grid(real_img, normalize = True)
        decoded_img , _ = self.model(real_img)
        decoded_img = decoded_img.view(-1, 3, 128,128).to(device).detach()
        decoded_grid = \
            torchvision.utils.make_grid(decoded_img, normalize = True)
        decoded_grid = decoded_grid.to("cpu").permute(1,2,0)
        real_grid = real_grid.to("cpu").permute(1,2,0)
        ax[0].imshow(decoded_grid)
        ax[0].title.set_text("Decoded Image")
        ax[1].imshow(real_grid)
        ax[1].title.set_text("Real Image")
        plt.show()
        
    def VQVAE_discrete(self, coords):
        #generate the discrete format of the image
        _, ax = plt.subplots(1,2)
        encoder = self.model.get_encoder()
        VQ = self.model.get_VQ()
        
        real_img = self.dataset[coords[0]][coords[1]]
        real_img = real_img.view(-1, 3, 128, 128).to(device).detach()
        real_grid = torchvision.utils.make_grid(real_img, normalize = True)
        encoded = encoder(real_img)
        
        encoded = encoded.permute(0, 2, 3, 1)
        flat_encoded  = encoded.view(-1, VQ.embedding_dim)
        _, lookup_indices = VQ.argmin_indices(flat_encoded)
        lookup_indices = lookup_indices.view(32,32).to(device)
        
        #indices_grid = torchvision.utils.make_grid(lookup_indices, normalize = True)
        
        real_grid = real_grid.to("cpu").permute(1,2,0)
        #indices_grid = indices_grid.to("cpu").permute(1,2,0)
        
        ax[0].imshow(lookup_indices.cpu().detach().numpy())
        ax[0].title.set_text("Latent Representation")
        ax[1].imshow(real_grid)
        ax[1].title.set_text("Real Representation")
        
        plt.show()
  
        