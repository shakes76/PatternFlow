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
from CustomLayers import *


## Todo: move this to customlayers
class UNetBlock(kr.layers.Layer) :
    def __init__(self, outputSize, upBlock = False, upSampler = ConvUpsample, downSampler = ConvDownsample) :
        super().__init__()
        
        self.conv1 = kr.layers.Conv2D(outputSize, 3, padding = "same", activation = kr.layers.ReLu)
        self.bNor1 = keras.Layers.BatchNormalization()
        self.conv2 = kr.layers.Conv2D(outputSize, 3, padding = "same", activation = kr.layers.ReLu)
        self.bNorm2 = keras.Layers.BatchNormalization()
        
        # If upsampling occurs in block
        if (upBlock) :
            self.transformer = upSampler(outputSize)
        else :
            self.transformer = downSampler(outputSize)
    
    def call(self, inputs) :
        imageInput = inputs[0]
        timeEmbedding = inputs[1]
        
        
        x = self.conv1(imageInput) 
        x = self.bNorm1(x) + timeEmedding
        x = self.conv2(x)
        x = self.bNorm2(x)
        
        transformed = self.transformer(x)
        return (x, transformed)

        
class UNet(kr.layers.Layer):
    def __init__(self, blocks = [64, 128, 256, 512], bottleNeck = 1024) :
        super().__init__()
        
        self.layerCount = len(blocks)
        self.finalLayer = kr.layers.Conv2D(filters = 1, kernel_size = 1, strides = 1, activation = kr.activations.relu)
        self.bottleNeck = UNetBlock(bottleNeck, upBlock = True)
        
        
        self.downBlocks = []
        self.upBlocks = []
        
        for b in blocks :
            self.downBlocks.append(UNetBlock(b, upBlock = False))
            self.upBlocks.insert(0, UNetBlock(b, upBlock = True))
            
    def call(self, image, timeEmbedding) :
        lastBlock = image
        skipConnections = []
        
        for i in range(self.layerCount) :
            skip, lastBlock = self.downBlocks[i]((lastBlock, timeEmbdedding))
            skipConnections.insert(0, skip)
            
        _, lastBlock = self.bottleNeck(lastBlock, timeEmbedding)
        
        for i in range(self.layerCount) :
            lastBlock = keras.concatenate((skipConnections[i], lastBlock))
            _, lastBlock = self.upBlocks[i]((image, timeEmbdedding))
            
        return self.finalLayer(lastBlock)
  

#class StableDiffusionModel(kr.Model) :
    """
    Implementation of a Stable Diffusion Model. 
    
    Uses the AutoEncoder class to generate and decode from the latent space.
    Uses the DiffusionModel class to generate new data from noised latent space.
    
    """
    
    """
    
############################### SIMPLY ADD TIME EMBEDDING AT EACH STEP!!!! ####
    def __init__(self):
        super().__init__()
        
        # Input Image
        self.input =
        self.timeEmbedding = kr.layers.Input()
        
    def diffusionSchedule() :
        
        # linear diffusion schedule:
        return tf.linspace(0.0001, 0.9999)
        
    def denoise(self) :
        
    
        
    def __addNoise(self, images) :
        
    def train_step(self, images) :
        
        # Channel first format:
        noise = tf.random.normal(shape=(batch_size, 3, image_size, image_size))
        
        
        with tf.GradientTape as tape:
            
"""
    
    
    


### AUTOENCODER PRE TRAINED, SIMILAR TO VQGAN PAPER
