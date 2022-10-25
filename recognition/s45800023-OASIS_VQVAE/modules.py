# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 12:11:40 2022

Modules file: Contains helper functions/Classes to be used in conjunction
with main scripts. 

@author: Jacob Barrie: s45800023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
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
        
## VQ-VAE ##
        
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

class VectorQuantizer(nn.Module):
    """
    Class implementing the Vector quantiser for the VQVAE
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

class VQVAE(nn.Module):
    """
    Class combining sub-modules into the full VQ-VAE model.
    """
    def __init__(self, num_hiddens, num_residual_hiddens, 
             num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(3, num_hiddens,
                                num_residual_hiddens)
        self.conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)

        self.vq = VectorQuantizer(num_embeddings, 
                                  embedding_dim,
                                  commitment_cost)
        
        self.decoder = Decoder(embedding_dim,
                               num_hiddens, 
                               num_residual_hiddens)
        
    def forward(self, x):
        out = self.encoder(x)
        out = self.conv(out)
        loss, quantized, perplexity, _ = self.vq(out)
        out_reconstruction = self.decoder(quantized)

        return loss, out_reconstruction
        
        
    
## DCGAN ##

class Discriminator(nn.Module):
    """
    Class implementing the discriminator for DCGAN
    """
    def __init__(self, channels, features):
        super(Discriminator, self).__init__()
        # 64 x 64 - input
        self.conv1 = nn.Conv2D(channels, features, kernel_size=4, stride=2, padding=1)
        self.leaky1 = nn.LeakyReLU(0.2)
        
        # Block 1 - 32x32 input
        self.conv2 = nn.Conv2D(features, features*2, kernel_size=4, stride=2, padding=1)
        self.batch1 = nn.BatchNorm2d(features*2)
        self.leaky2 = nn.LeakyReLU(0.2)
        
        # Block 2 - reduce to 1
        self.conv3 = nn.Conv2D(features*2, features*4, kernel_size=4, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(features*4)
        self.leaky3 = nn.LeakyReLU(0.2)
        
        # Block 3 - reduce to 32x32
        self.conv4 = nn.Conv2D(features*4, features*8, kernel_size=4, stride=2, padding=1)
        self.batch3 = nn.BatchNorm2d(features*4)
        self.leaky4 = nn.LeakyReLU(0.2)
        
        
        # Out - reduces to 1 dimension
        self.out = nn.Conv2D(features*8, 1, kernel_size=4, stride=2, padding=0)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.leaky1(out)
        out = self.conv2(out)
        out = self.batch1(out)
        out = self.leaky2(out)
        out = self.conv3(out)
        out = self.batch2(out)
        out = self.leaky3(out)
        out = self.conv4(out)
        out = self.batch3(out)
        out = self.leaky4(out)
        out = self.out(out)
        out = self.sigmoid(out)
        return out
        
class Generator(nn.Module):
    """
    Class implementing the generator for DCGAN
    """
    def __init__(self, channels_noise, channels, features):
        super(Generator, self).__init__()     
        # Block 1 - input noise
        self.convT1 = nn.ConvTranspose2d(channels_noise, features*16, kernel_size=4, stride=1, padding=0)
        self.batch1 = nn.BatchNorm2d(features*16)
        self.relu1 = nn.ReLU()
        
        # Block 2
        self.convT2 = nn.ConvTranspose2d(features*16, features*8, kernel_size=4, stride=1, padding=1)
        self.batch2 = nn.BatchNorm2d(features*8)
        self.relu2 = nn.ReLU()
        
        # Block 3
        self.convT3 = nn.ConvTranspose2d(features*8, features*4, kernel_size=4, stride=1, padding=1)
        self.batch3 = nn.BatchNorm2d(features*4)
        self.relu3 = nn.ReLU()
        
        # Block 4
        self.convT4 = nn.ConvTranspose2d(features*4, features*2, kernel_size=4, stride=1, padding=1)
        self.batch4 = nn.BatchNorm2d(features*8)
        self.relu4 = nn.ReLU()
        
        self.convT5 = nn.ConvTranspose2d(features*2, features, kernel_size=4, stride=1, padding=1)
        self.tanH = nn.Tanh()
        
    def forward(self, x):
        out = self.convT1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        out = self.convT2(out)
        out = self.batch2(out)
        out = self.relu2(out)
        out = self.convT3(out)
        out = self.batch3(out)
        out = self.relu3(out)
        out = self.convT4(out)
        out = self.batch4(out)
        out = self.relu4(out)
        out = self.convT5(out)
        out = self.tanH(out)
        return out
        
        
def trainDCGAN():
    """
    Class to hold hyper-parameters and implement the training process for 
    DCGAN.
    
    Returns
    -------
    None.

    """
    def __init__(self, Discriminator, Generator, data):
        """
        Initailize hyper-parameters and pass in models/data.
        
        Parameters
        ----------
        Discriminator : nn.Module
            Discriminator network
        Generator : nn.Module
            Generator network
        data : String
            data path
        Returns
        -------
        None.

        """
        self.Discriminator = Discriminator
        self.Generator = Generator
        self.data = data
        self.batch_size = 32
        self.epochs = 10
        self.loss = nn.BCELoss()
        self.lr = 0.0002
        self.optimizer_g = torch.optim.Adam(self.Generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.optimizer_d = torch.optim.Adam(self.Discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
    
    def quantizeData(self):
        """
        We are using the DCGAN to generate codebooks indices in the latent space.
        Thus when training the model we need to pass in 
        """
        return
    
    def train(self):
        return