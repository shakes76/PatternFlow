# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 07:46:19 2021

@author: jmill
"""

#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam


"""
Improved UNet Architecture:
    
    This model is based out of the improved 
    
"""

class ImprovedUnet():

    def create_model(self, image_size):
        input_size = (image_size, image_size, 3)
        #Image segmentation predominantly uses normally distributed values around 0 as the kernel_initializer.
        ini = "he_normal"
        
        ###Pseudocode for Improved U-Net
        inputs = Input(shape = input_size)
        
        #Conv layer 1
        conv1 = Conv2D(16,3, activation = "relu", padding = "same", kernel_initializer = ini)(inputs)
        conv1 = Conv2D(16,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv1)
        #Conv layer 1 - context module All context modules per the paper are a LeakyReLU layer, 2 conv layers and then a dropout layer.
        conv1 = LeakyReLU()(conv1)
        conv1 = Conv2D(16,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv1)
        conv1 = Conv2D(16,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv1)
        conv1 = Dropout(0.3)(conv1)
        conv1 = Normalization()(conv1)
        
        #Conv Layer 2
        conv2 = Conv2D(32, 3, 2, activation = "relu", padding = "same", kernel_initializer = ini)(conv1)
        #Conv Layer 2 - Context Module
        conv2 = Normalization()(conv2)
        conv2 = LeakyReLU()(conv2)
        conv2 = Conv2D(32,3, activation = "relu", padding= "same", kernel_initializer = ini)(conv2)
        conv2 = Conv2D(32,3, activation = "relu", padding= "same", kernel_initializer = ini)(conv2)
        conv2 = Dropout(0.3)(conv2)
        conv2 = Normalization()(conv2)
        
        #Conv Layer 3
        conv3 = Conv2D(64,3,2, activation = "relu", padding = "same", kernel_initializer = ini)(conv2)
        #Conv Layer 3 - Context Module
        conv3 = Normalization()(conv3)
        conv3 = LeakyReLU()(conv3)
        conv3 = Conv2D(64,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv3)
        conv3 = Conv2D(64,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv3)
        conv3 = Dropout(0.3)(conv3)
        conv3 = Normalization()(conv3)
        
        #Conv layer 4
        conv4 = Conv2D(128,3,2, activation = "relu", padding = "same", kernel_initializer = ini)(conv3)
        #Conv layer 4 - Context Module
        conv4 = Normalization()(conv4)
        conv4 = LeakyReLU()(conv4)
        conv4 = Conv2D(128,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv4)
        conv4 = Conv2D(128,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv4)
        conv4 = Dropout(0.3)(conv4)
        conv4 = Normalization()(conv4)
        
        #Conv layer 5
        conv5 = Conv2D(256,3,2, activation = "relu", padding = "same", kernel_initializer = ini)(conv4)
        #Conv layer 5 - Context Module
        conv5 = Normalization()(conv5)
        conv5 = LeakyReLU()(conv5)
        conv5 = Conv2D(256,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv5)
        conv5 = Conv2D(256,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv5)
        conv5 = Dropout(0.3)(conv5)
        conv5 = Normalization()(conv5)
        
        
        #Conv layer 6 - > upscale
        conv6 = Conv2D(128,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv5)
        conv6 = UpSampling2D((2,2))(conv6)
        
        merge1 = concatenate([conv4,conv6], axis = 3)
        
        #Conv layer 7 - Localisation/upscale
        conv7 = Normalization()(merge1)
        conv7 = LeakyReLU()(conv7)
        
        conv7 = Conv2D(128,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv7)
        conv7 = Conv2D(128,1, activation = "relu", padding = "same", kernel_initializer = ini)(conv7)
        conv7 = Normalization()(conv7)
        conv7 = LeakyReLU()(conv7)
        
        #Conv layer 8 -> upscale
        conv8 = Conv2D(64,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv7)
        conv8 = UpSampling2D((2,2))(conv8)
        merge2 = concatenate([conv3,conv8], axis = 3)
        
        #conv layer 9 -> localisation/upscale
        conv9 = Normalization()(merge2)
        conv9 = LeakyReLU()(conv9)
        conv9 = Conv2D(64,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv9)
        conv9 = Conv2D(64,1, activation = "relu", padding = "same", kernel_initializer = ini)(conv9)
        conv9 = Normalization()(conv9)
        conv9 = LeakyReLU()(conv9)
        
        #Conv layer 10 -> upscale
        conv10 = Conv2D(32,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv9)
        conv10 = UpSampling2D((2,2))(conv10)
        merge3 = concatenate([conv2,conv10], axis =3)
        
        #conv layer 11 - > localisation/upscale
        conv11 = Normalization()(merge3)
        conv11 = LeakyReLU()(conv11)
        conv11 = Conv2D(32,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv11)
        conv11 = Conv2D(32,1, activation = "relu", padding = "same", kernel_initializer = ini)(conv11)
        conv11 = Normalization()(conv11)
        conv11 = LeakyReLU()(conv11)
        
        #Conv layer 12 -> localisation/upscale
        conv12 = Conv2D(16,3, activation = "relu", padding = "same", kernel_initializer = ini)(conv11)
        conv12 = UpSampling2D((2,2))(conv12)
        merge4 = concatenate([conv1, conv12], axis = 3)
        #Conv layer 13 
        conv13 = Conv2D(32,3,activation = "relu", padding = "same", kernel_initializer = ini)(conv12)
        #Output layer -> Has been left as 256, as it may allow for other segmentations with different classes.
        output = Conv2D(256,1, activation = "softmax")(conv13)
        
        #Generate the model and compile it
        model = tf.keras.Model(inputs = inputs, outputs = output)
        model.compile(optimizer = Adam(learning_rate = 0.0001), loss = tf.keras.losses.SparseCategoricalCrossentropy(), metrics = ['accuracy'])
        
        #return the model.
        return model
    
    
    

    