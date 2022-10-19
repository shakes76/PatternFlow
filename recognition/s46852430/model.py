# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:55:06 2022

@author: eudre
"""
import tensorflow as tf
from tensorflow.keras import models, layers, Model, Input
from tensorflow.keras.layers import Activation, Add, Conv3D
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU, concatenate
from tensorflow.keras.layers import UpSampling2D, MaxPooling3D, Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose
tf.random.Generator = None

def conv_block(input_matrix, num_filter):
  X = Conv3D(num_filter,kernel_size = 3, strides=(1,1,1),padding='same')(input_matrix)
  X = BatchNormalization()(X)
  
  X = Activation('relu')(X)
  X = Conv3D(num_filter,kernel_size = 3,strides=(1,1,1),padding='same')(X)
  
  X = BatchNormalization()(X)
  
  X = Activation('relu')(X)
  
  return X


def modified_UNET(input_shape, num_filter, dropout = 0.3):

  inputs = Input(input_shape)
  
  #Encode
  c1 = conv_block(inputs,num_filter)
  p1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(c1)
  p1 = Dropout(dropout)(p1)
  
  c2 = conv_block(p1,num_filter*2)
  p2 = MaxPooling3D(pool_size=(2, 2, 2) ,strides=2)(c2)
  p2 = Dropout(dropout)(p2)

  c3 = conv_block(p2,num_filter*4);
  p3 = MaxPooling3D(pool_size=(2, 2, 2) ,strides=2)(c3)
  p3 = Dropout(dropout)(p3)
  
  c4 = conv_block(p3,num_filter*8);
  p4 = MaxPooling3D(pool_size=(2, 2, 2) ,strides=2)(c4)
  p4 = Dropout(dropout)(p4)
  
  c5 = conv_block(p4,num_filter*16);

  # Decode    
  u6 = Conv3DTranspose(num_filter*8, (3, 3, 3), strides=(2, 2, 2), padding='same')(c5);
  u6 = concatenate([u6,c4]);
  c6 = conv_block(u6,num_filter*8)
  c6 = Dropout(dropout)(c6)
  u7 = Conv3DTranspose(num_filter*4,(3, 3, 3),strides = (2, 2, 2) , padding= 'same')(c6);

  u7 = concatenate([u7,c3]);
  c7 = conv_block(u7,num_filter*4)
  c7 = Dropout(dropout)(c7)
  u8 = Conv3DTranspose(num_filter*2,(3, 3, 3),strides = (2, 2, 2) , padding='same')(c7);
  u8 = concatenate([u8,c2]);

  c8 = conv_block(u8,num_filter*2)
  c8 = Dropout(dropout)(c8)
  u9 = Conv3DTranspose(num_filter,(3, 3, 3),strides = (2, 2, 2) , padding='same')(c8);

  u9 = concatenate([u9,c1]);

  c9 = conv_block(u9,num_filter)
  
  # Output
  outputs = Conv3D(4, (1, 1, 1), activation='softmax')(c9)

  model = Model(inputs, outputs)

  return model
