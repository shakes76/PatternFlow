# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 16:29:31 2022

@author: blobf
"""
import torch
import torchvision

import modules

import matplotlib.pyplot as plt


device = "cuda" if torch.cuda.is_available() else "cpu"

"""
Class dedicated the Visualisation of VQVAE image comparison

Compares original and encoded/decoded images
"""
class VQVAE_Visualise():
    """
    Parameters:
        VQVAE_path -> path to the .pt VQVAE model
        num_embeddings -> number of embeddings of the VQVAE
        latent_dim -> latent dimension of the VQVAE
        data -> dataset provided for visualisation
    
    """
    def __init__(self, VQVAE_path, num_embeddings, latent_dim, data):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        #load the trained model from a pt file
        model = modules.VQVAE(num_embeddings, latent_dim, 0.25)
        state_dict = torch.load(VQVAE_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        self.model = model
        
        
        #load the image data
        self.dataset = data
    """
    Visualise the comparison between the original image and image that went 
    through VQVAE.
    
    Parameters:
        coords -> coordinates of the picture specified from the dataset 
        provided
    
    """
    def visualise_VQVAE(self, coords):
        
        real_img = self.dataset[coords[0]][coords[1]]
        _, ax = plt.subplots(1,2)
        
        real_img = real_img.view(-1, 3, 128,128).to(device).detach()
        real_grid = torchvision.utils.make_grid(real_img, normalize = True)
        decoded_img , _ = self.model(real_img) # put real_img through VQVAE
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
        
    """
    Visualise the comparison between the original image and its latent
    representation
    
    Parameters:
        coords -> coordinates of the specified image in data set (choose an 
                                                                  image, like 
                                                                  (0,0))
    """
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
        # encode the image and get its codebook representation
        flat_encoded  = encoded.view(-1, VQ.embedding_dim)
        _, lookup_indices = VQ.argmin_indices(flat_encoded)
        lookup_indices = lookup_indices.view(32,32).to(device)
        
        
        real_grid = real_grid.to("cpu").permute(1,2,0)
        
        ax[0].imshow(lookup_indices.cpu().detach().numpy())
        ax[0].title.set_text("Latent Representation")
        ax[1].imshow(real_grid)
        ax[1].title.set_text("Real Representation")
        
        plt.show()
  
        