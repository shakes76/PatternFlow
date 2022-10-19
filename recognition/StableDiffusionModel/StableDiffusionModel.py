# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:33:35 2022

@author: Danie
"""

import tensorflow.keras as kr
import tensorflow as tf
import matplotlib as plt
import numpy as np

from keras import layers



#TODO: If time permits, unhardcode this later
class UNet(kr.layers.Layer):
    def __init__(self) :
    # Downsampling Process
        self.convA1 = kr.layers.Conv2D(64, 3, padding = "same", activation = kr.activations.ReLu)
        self.convA2 = kr.layers.Conv2D(64, 3, padding = "same", activation = kr.activations.ReLu)
        self.downSampleA1 = kr.layers.MaxPooling2D((2,2))
        
        self.convB1 = kr.layers.Conv2D(128, 3, padding = "same", activation = kr.activations.ReLu)
        self.convB2 = kr.layers.Conv2D(128, 3, padding = "same", activation = kr.activations.ReLu)
        self.downSampleB1 = kr.layers.MaxPooling2D((2,2))
        
        self.convC1 = kr.layers.Conv2D(256, 3, padding = "same", activation = kr.activations.ReLu)
        self.convC2 = kr.layers.Conv2D(256, 3, padding = "same", activation = kr.activations.ReLu)
        self.downSampleC1 = kr.layers.MaxPooling2D((2,2))
        
        self.convD1 = kr.layers.Conv2D(512, 3, padding = "same", activation = kr.activations.ReLu)
        self.convD2 = kr.layers.Conv2D(512, 3, padding = "same", activation = kr.activations.ReLu)
        self.downSampleD1 = kr.layers.MaxPooling2D((2,2))
        
    # Lowest layer / start of upsampling process
        self.convE1 = kr.layers.Conv2D(1024, 3, padding = "same", activation = kr.activations.ReLu)
        self.convE2 = kr.layers.Conv2D(1024, 3, padding = "same", activation = kr.activations.ReLu)
        self.upSampleE1 = kr.layers.Conv2DTranspose(512, 3, 2, padding = "same")
        
        
        self.convF1 = kr.layers.Conv2D(512, 3, padding = "same", activation = kr.activations.ReLu)
        self.convF2 = kr.layers.Conv2D(512, 3, padding = "same", activation = kr.activations.ReLu)
        self.upSampleF1 = kr.layers.Conv2DTranspose(256, 3, 2, padding = "same")
        
        self.convG1 = kr.layers.Conv2D(256, 3, padding = "same", activation = kr.activations.ReLu)
        self.convG2 = kr.layers.Conv2D(256, 3, padding = "same", activation = kr.activations.ReLu)
        self.upSampleG1 = kr.layers.Conv2DTranspose(128, 3, 2, padding = "same")
        
        self.convH1 = kr.layers.Conv2D(128, 3, padding = "same", activation = kr.activations.ReLu)
        self.convH2 = kr.layers.Conv2D(128, 3, padding = "same", activation = kr.activations.ReLu)
        self.upSampleH1 = kr.layers.Conv2DTranspose(64, 3, 2, padding = "same")
        
        self.convI1 = kr.layers.Conv2D(64, 3, padding = "same", activation = kr.activations.ReLu)
        self.convI2 = kr.layers.Conv2D(64, 3, padding = "same", activation = kr.activations.ReLu)
        self.finalConv = kr.layers.Conv2D(2, 1, padding = "same", activation = kr.activations.ReLu)
        
    def call(self, inputs):
        
        
        a = self.convA1(inputs)
        a = self.convA2(a)
        
        b = self.downSampleA1(a)
        b = self.convB1(b)
        b = self.convB2(b)
        
        c = self.downSampleB1(b)
        c = self.convC1(c)
        c = self.convC2(c)
        
        
        d = self.downSampleC1(c)
        d = self.convD1(d)
        d = self.convD2(d)
        
        e = self.downSampleD1(d)
        e = self.convE1(e)
        e = self.convE2(e)
        
        f = self.upSampleE1(e)
        f = kr.layers.concatenate((d,f))
        f = self.convF1(f)
        f = self.convF2(f)
        
        g = self.upSampleF1(f)
        g = kr.layers.concatenate((c,g))
        g = self.convG1(g)
        g = self.convG2(g)
        
        h = self.upSampleG1(g)
        h = kr.layers.concatenate((b,h))
        h = self.convG1(h)
        h = self.convG2(h)
        
        i = self.upSampleH1(h)
        i = kr.layers.concatenate((a,i))
        i = self.convG1(h)
        i = self.convG2(h)
        
        return self.finalConv(i)
        
        
        
        
    
    


class StableDiffusionModel(kr.Model) :
    """
    Implementation of a Stable Diffusion Model. 
    
    Uses the AutoEncoder class to generate and decode from the latent space.
    Uses the DiffusionModel class to generate new data from noised latent space.
    
    """
    
    
    


### AUTOENCODER PRE TRAINED, SIMILAR TO VQGAN PAPER
