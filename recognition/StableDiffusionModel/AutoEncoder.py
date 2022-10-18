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

class Encoder(kr.Sequential) :
    def __init__(self, inputSize, activation = kr.activations.swish, downSampleLayer = ConvDownsample) :
        super().__init__(
            [
            # Block downsampling by factor of 2
                ResidualNetBlock(1, 64, 3, activation = activation),
                ResidualNetBlock(64, 64, 3, activation = activation),
                ResidualNetBlock(64, 64, 3, activation = activation), 
                downSampleLayer(64),
            
            # Block downsampling by factor of 2
                ResidualNetBlock(64, 64, 3, activation = activation),  
                ResidualNetBlock(64, 64, 3, activation = activation), 
                downSampleLayer(64),
            
            # Block downsampling by factor of 2
                ResidualNetBlock(64, 64, 3, activation = activation),   
                ResidualNetBlock(64, 64, 3, activation = activation),   
                downSampleLayer(64),
            
            # Reducing filter channel back to 1
                ResidualNetBlock(64, 64, 3, activation = activation),   
                ResidualNetBlock(64, 64, 3, activation = activation),  
                ResidualNetBlock(64, 1, 3, activation = activation)
            ])
        
        
################################################ CONSIDER DECONVOLUTION  ################################################

class Decoder(kr.Sequential) :
    def __init__(self, latentSpaceSize, outputSize, activation = kr.activations.swish, upSampleLayer = kr.layers.UpSampling2D) :
        super().__init__(
            [    
            # Block upsampling by factor of 2
                ResidualNetBlock(1, 64, 3, activation = activation),
                ResidualNetBlock(64, 64, 3, activation = activation),
                ResidualNetBlock(64, 64, 3, activation = activation),   
                upSampleLayer(),
            
            # Block upsampling by factor of 2
                ResidualNetBlock(64, 64, 3, activation = activation),   
                ResidualNetBlock(64, 64, 3, activation = activation),   
                upSampleLayer(),
            
            # Block upsampling by factor of 2
                ResidualNetBlock(64, 64, 3, activation = activation),   
                ResidualNetBlock(64, 64, 3, activation = activation),   
                upSampleLayer(),
            
            # Reducing filter channel back to 1
                ResidualNetBlock(64, 64, 3, activation = activation),   
                ResidualNetBlock(64, 64, 3, activation = activation),  
                ResidualNetBlock(64, 1, 3, activation = activation)
            ])
                                 
class AutoEncoder(kr.Model) :
    """
    Implementaion of an autoencoder model.
    
    """
    
    def __init__(self, inputSize, latentSpaceSize, activation = kr.activations.swish) :
        super().__init__()
        
        self.inputSize = inputSize
        self.latentSpaceSize = latentSpaceSize
        
        self.encoder = self.__buildEncoderLayers(inputSize, activation=activation)
        self.decoder = self.__buildDecoderLayers(latentSpaceSize, inputSize, activation=activation)
        
    def call(self, inputs) :
        x = self.encoder(inputs)
        return self.decoder(x)
        
    def __buildEncoderLayers(self, inputSize, activation=kr.activations.swish) : 
        return Encoder(inputSize, activation=activation)
        
        
    def __buildDecoderLayers(self, latentSpaceSize, outputSize, activation=kr.activations.swish) :
        return Decoder(latentSpaceSize, outputSize, activation=activation)
        

    def buildEncoder(self) :
        newInput = kr.Input(self.inputSize)
        return kr.models.Model(newInput, self.encoder(newInput)) ####################################################
    
    def buildDecoder(self) :
        newLatent = kr.Input(self.latentSpaceSize)
        return kr.models.Model(newLatent, self.decoder(newLatent)) ####################################################

