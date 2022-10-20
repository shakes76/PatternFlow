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
        
        self.norm1 = kr.layers.BatchNormalization()
        
    def call(self, inputs) :
        return self.norm1(self.conv1(inputs))
    

class ConvUpsample(kr.layers.Layer) :
    def __init__(self, outputSize):
        super().__init__()
        self.transform = kr.layers.Conv2DTranspose(outputSize, 3, 2, padding="same")
        self.norm1 = kr.layers.BatchNormalization()
        
    def call(self, inputs) :
        return self.norm1(self.transform(inputs))
        
        
### Used in AutoEncoder / Unet
class ResidualNetBlock(kr.layers.Layer) :
    def __init__(self, inputDim, outputDim, kernelSize, normLayers = True, activation = None, epsilon = 1e-4) :
        super().__init__()
        
        self.__isNormed = normLayers
        
        self.conv1 = ZeroPaddedConv2D(outputDim, kernelSize, activation = activation)
        #self.conv2 = ZeroPaddedConv2D(outputDim, kernelSize, activation = activation)
        self.norm1 = kr.layers.BatchNormalization()
        #self.norm2 = kr.layers.BatchNormalization(epsilon = epsilon)
        
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
            x = x + self.skip1(inputs)
            #x = self.conv2(x)
            #x = self.norm2(x)
            
            return self.norm1(x)
        else :
            # If groupNorm layers are not enabled
            x = self.conv1(inputs)
            x = self.conv2(x)
            
            return x + self.skip1(inputs)