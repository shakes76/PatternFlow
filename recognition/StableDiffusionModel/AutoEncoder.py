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
    def __init__(self, inputSize, activation = kr.activations.swish, downSampleLayer = ConvDownsample, normLayers = True) :
        super().__init__(
            [
            # Block downsampling by factor of 2
                ResidualNetBlock(1, 64, 3, activation = activation, normLayers = normLayers),
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers), 
                downSampleLayer(64),
            
            # Block downsampling by factor of 2
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),  
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers), 
                downSampleLayer(64),
            
            # Block downsampling by factor of 2
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),   
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),   
                downSampleLayer(64),
            
            # Reducing filter channel back to 1
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),   
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),  
                ResidualNetBlock(64, 1, 3, activation = activation, normLayers = normLayers)
            ])
        
        
################################################ CONSIDER DECONVOLUTION  ################################################

class Decoder(kr.Sequential) :
    def __init__(self, latentSpaceSize, outputSize, activation = kr.activations.swish, upSampleLayer = kr.layers.UpSampling2D, normLayers = True) :
        super().__init__(
            [    
            # Block upsampling by factor of 2
                ResidualNetBlock(1, 64, 3, activation = activation, normLayers = normLayers),
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),   
                upSampleLayer(),
            
            # Block upsampling by factor of 2
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),   
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),   
                upSampleLayer(),
            
            # Block upsampling by factor of 2
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),   
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),   
                upSampleLayer(),
            
            # Reducing filter channel back to 1
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),   
                ResidualNetBlock(64, 64, 3, activation = activation, normLayers = normLayers),  
                ResidualNetBlock(64, 1, 3, activation = activation, normLayers = normLayers)
            ])
                                 
class AutoEncoder(kr.Model) :
    """
    Implementaion of an autoencoder model.
    
    """
    
    def __init__(self, inputSize, latentSpaceSize, activation = kr.activations.swish, normLayers = True) :
        super().__init__()
        
        self.inputSize = inputSize
        self.latentSpaceSize = latentSpaceSize
        
        self.encoder = self.__buildEncoderLayers(inputSize, activation=activation, normLayers = normLayers)
        self.decoder = self.__buildDecoderLayers(latentSpaceSize, inputSize, activation=activation, normLayers = normLayers)
        
    def call(self, inputs) :
        x = self.encoder(inputs)
        return self.decoder(x)
        
    def __buildEncoderLayers(self, inputSize, activation=kr.activations.swish, normLayers = True) : 
        return Encoder(inputSize, activation=activation, normLayers = normLayers)
        
        
    def __buildDecoderLayers(self, latentSpaceSize, outputSize, activation=kr.activations.swish, normLayers = True) :
        return Decoder(latentSpaceSize, outputSize, activation=activation, normLayers = normLayers)
        

    def buildEncoder(self) :
        newInput = kr.Input(self.inputSize)
        return kr.models.Model(newInput, self.encoder(newInput)) ####################################################
    
    def buildDecoder(self) :
        newLatent = kr.Input(self.latentSpaceSize)
        return kr.models.Model(newLatent, self.decoder(newLatent)) ####################################################
