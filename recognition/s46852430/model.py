# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 12:55:06 2022

@author: eudre
"""
import tensorflow as tf
from tensorflow.keras import models, layers, Model, Input
from tensorflow.keras.layers import Activation, Add, Conv2D, Conv3D
from tensorflow.keras.layers import BatchNormalization, Dropout, LeakyReLU, concatenate
from tensorflow.keras.layers import UpSampling2D, MaxPooling3D, Conv2D, Conv2DTranspose,Conv3D,Conv3DTranspose
tf.random.Generator = None

def conv_block(input_matrix, num_filter, kernel_size, batch_norm):
  X = Conv3D(num_filter,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(input_matrix)
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)

  X = Conv3D(num_filter,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(X)
  if batch_norm:
    X = BatchNormalization()(X)
  
  X = Activation('relu')(X)
  
  return X


def modified_UNET(input_img, dropout = 0.2, batch_norm = True):
#Encode
  c1 = conv_block(input_img,8,3,batch_norm)
  p1 = MaxPooling3D(pool_size=(2, 2, 2), strides=2)(c1)
  p1 = Dropout(dropout)(p1)
  
  c2 = conv_block(p1,16,3,batch_norm);
  p2 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c2)
  p2 = Dropout(dropout)(p2)

  c3 = conv_block(p2,32,3,batch_norm);
  p3 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c3)
  p3 = Dropout(dropout)(p3)
  
  c4 = conv_block(p3,64,3,batch_norm);
  p4 = MaxPooling3D(pool_size=(2,2,2) ,strides=2)(c4)
  p4 = Dropout(dropout)(p4)
  
  c5 = conv_block(p4,128,3,batch_norm);

# Decode
  u6 = Conv3DTranspose(64, (3,3,3), strides=(2, 2, 2), padding='same')(c5);
  u6 = concatenate([u6,c4]);
  c6 = conv_block(u6,64,3,batch_norm)
  c6 = Dropout(dropout)(c6)
  u7 = Conv3DTranspose(32,(3,3,3),strides = (2,2,2) , padding= 'same')(c6);

  u7 = concatenate([u7,c3]);
  c7 = conv_block(u7,32,3,batch_norm)
  c7 = Dropout(dropout)(c7)
  u8 = Conv3DTranspose(16,(3,3,3),strides = (2,2,2) , padding='same')(c7);
  u8 = concatenate([u8,c2]);

  c8 = conv_block(u8,16,3,batch_norm)
  c8 = Dropout(dropout)(c8)
  u9 = Conv3DTranspose(8,(3,3,3),strides = (2,2,2) , padding='same')(c8);

  u9 = concatenate([u9,c1]);

  c9 = conv_block(u9,8,3,batch_norm)
  outputs = Conv3D(4, (1, 1,1), activation='softmax')(c9)
  print("!!!!!!!!!!!!!!!!!!!")
  print(outputs.shape)
  model = Model(inputs=input_img, outputs=outputs)

  return model
