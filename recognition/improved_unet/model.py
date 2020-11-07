# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 11:55:46 2020

@author: danny
"""

def unet_model(output_channels, f=64):
    inputs = tf.keras.layers.Input(shape=(256,256,1))
    
    d1 = tf.keras.layers.Conv2D(1*f, 3, padding='same', activation='relu')(inputs)
    d1 = tf.keras.layers.Conv2D(1*f, 3, padding='same', activation='relu')(d1)
    
    d2 = tf.keras.layers.MaxPooling2D()(d1)
    d2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(d2)
    d2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(d2)
    
    d3 = tf.keras.layers.MaxPooling2D()(d2)
    d3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(d3)
    d3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(d3)
    
    d4 = tf.keras.layers.MaxPooling2D()(d3)
    d4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(d4)
    d4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(d4)
    
    d5 = tf.keras.layers.MaxPooling2D()(d4)
    d5 = tf.keras.layers.Conv2D(16*f, 3, padding='same', activation='relu')(d5)
    d5 = tf.keras.layers.Conv2D(16*f, 3, padding='same', activation='relu')(d5)
    
    u4 = tf.keras.layers.UpSampling2D()(d5)
    u4 = tf.keras.layers.concatenate([u4, d4])
    u4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(u4)
    u4 = tf.keras.layers.Conv2D(8*f, 3, padding='same', activation='relu')(u4)
    
    u3 = tf.keras.layers.UpSampling2D()(u4)
    u3 = tf.keras.layers.concatenate([u3, d3])
    u3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(u3)
    u3 = tf.keras.layers.Conv2D(4*f, 3, padding='same', activation='relu')(u3)
    
    u2 = tf.keras.layers.UpSampling2D()(u3)
    u2 = tf.keras.layers.concatenate([u2, d2])
    u2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(u2)
    u2 = tf.keras.layers.Conv2D(2*f, 3, padding='same', activation='relu')(u2)
    
    u1 = tf.keras.layers.UpSampling2D()(u2)
    u1 = tf.keras.layers.concatenate([u1, d1])
    u1 = tf.keras.layers.Conv2D(1*f, 3, padding='same', activation='relu')(u1)
    u1 = tf.keras.layers.Conv2D(1*f, 3, padding='same', activation='relu')(u1)
    
    outputs = tf.keras.layers.Conv2D(output_channels, 1, activation='softmax')(u1)
    return tf.keras.Model(inputs=inputs, outputs = outputs)