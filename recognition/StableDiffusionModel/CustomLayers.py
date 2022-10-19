# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:33:35 2022

@author: Danie
"""

import tensorflow.keras as kr
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib as plt
import numpy as np


### Unet architecture: 

##################################### PROBABLY USE SWISH AS ACTIV FUNCTION OVER RELU  //// LOOK INTO GEGLU
### Layers needed

    #### Padded Convolution layer (st. inputDim = outputDim)
    
class ZeroPaddedConv2D(kr.layers.Layer) :
    def __init__(self, filters, kernelSize =3, stride = (1,1), padding = (1,1), activation = None) :
        super().__init__()
        
        self.padd = kr.layers.ZeroPadding2D(padding = padding)
        self.conv = kr.layers.Conv2D(filters, kernelSize, stride, activation = activation)
        
    def call(self, inputs) :
        return self.conv(self.padd(inputs))
        
    
### Used in UNET // ################################# Used as alternative to MaxPool
class ConvDownsample(kr.layers.Layer) :
    def __init__(self, outputLayers, activation = None):
        super().__init__()
        
        self.conv1 = ZeroPaddedConv2D(outputLayers, 
                                         kernelSize = 3, 
                                         stride = (2,2), 
                                         padding = (1,1), 
                                         activation = activation)
        
    def call(self, inputs) :
        return self.conv1(inputs)
    
## Todo: make deconvolution for upsampling, see below:
class ConvUpsample(kr.layers.Layer) :
    def __init__(self, outputSize):
        super().__init__()
        self.transform = kr.layers.Conv2DTranspose(outputSize, 3, 2, padding="same")
        
    def call(self, inputs) :
        return self.transform(inputs)
        
        
### Used in AutoEncoder / Unet
class ResidualNetBlock(kr.layers.Layer) :
    def __init__(self, inputDim, outputDim, kernelSize, normLayers = True, activation = None, epsilon = 1e-6) :
        super().__init__()
        
        self.__isNormed = normLayers
        
        self.conv1 = ZeroPaddedConv2D(outputDim, kernelSize, activation = activation)
        self.conv2 = ZeroPaddedConv2D(outputDim, kernelSize, activation = activation)
        self.norm1 = tfa.layers.GroupNormalization(epsilon = epsilon)
        self.norm2 = tfa.layers.GroupNormalization(epsilon = epsilon)
        
        if (inputDim != outputDim) :
            # If the input dimension is different from the output dimension 
            # Convolution with kenrel size 1 and outputDim many filters
            # is used to resize data
            self.skip1 = kr.layers.Conv2D(outputDim, 1)
        else :
            # Else, input is summed, like standard skip connection.
            self.skip1 = kr.layers.Lambda(lambda x : x)
        
        
    def call(self, inputs) :
        if (self.__isNormed) :
            # if groupNorm layers are enabled
            
            x = self.conv1(inputs)
            x = self.norm1(x)
            x = self.conv2(x)
            x = self.norm2(x)
            
            return x + self.skip1(inputs)
        else :
            # If groupNorm layers are not enabled
            x = self.conv1(inputs)
            x = self.conv2(x)
            
            return x + self.skip1(inputs)
        
        
    