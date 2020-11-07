# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 11:55:46 2020

@author: danny
"""
import tensorflow as tf
import zipfile
import glob

def context_module(input, filter):
    d1 = tf.keras.layers.BatchNormalization()(input)
    d1 = tf.keras.layers.Conv2D(filter, (3, 3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d1)
    d2 = tf.keras.layers.Dropout(0.3)(d1)
    d3 = tf.keras.layers.BatchNormalization()(d2)
    d3 = tf.keras.layers.Conv2D(filter, (3, 3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(d3)
    return d3

def upsampling_module(input, filter):
    u1 = tf.keras.layers.UpSampling2D(size=(2,2))(input)
    u2 = tf.keras.layers.Conv2D(filter, (3, 3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(u1)
    return u2

def localiztion_module(input, filter):
    c1 = tf.keras.layers.Conv2D(filter, (3, 3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(input)
    c2 = tf.keras.layers.Conv2D(filter, (3, 3), padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(c1)
    return c2

def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=(256,256,1))
    
    #Encoding
    con1 = tf.keras.layers.Conv2D(16, (3, 3),strides = 2 , padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(inputs)     
    con2 = context_module(con1, 16)
    add1 = tf.keras.layers.Add()([con1, con2])
    
    con3 = tf.keras.layers.Conv2D(32, (3, 3), strides = 2 ,padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(add1)     
    con4 = context_module(con3, 32)
    add2 = tf.keras.layers.Add()([con3, con4])
    
    con5 = tf.keras.layers.Conv2D(64, (3, 3), strides = 2 ,padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(add2)     
    con6 = context_module(con5, 64)
    add3 = tf.keras.layers.Add()([con5, con6])
    
    con7 = tf.keras.layers.Conv2D(128, (3, 3), strides = 2 , padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(add3)     
    con8 = context_module(con7, 128)
    add4 = tf.keras.layers.Add()([con7, con8])
    
    con9 = tf.keras.layers.Conv2D(256, (3, 3), strides = 2 , padding = 'same', activation=tf.keras.layers.LeakyReLU(alpha=0.01))(add4)     
    con10 = context_module(con9, 256)
    add5 = tf.keras.layers.Add()([con9, con10])
    

    
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(u1)
    model = tf.keras.Model(inputs=inputs, outputs = outputs)
    
    return model