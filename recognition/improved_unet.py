"""
COMP3710 Report 

This file contains context_module function, localization_module function, upsampling module function and improved unet mode

@author  Linyuzhuo Zhou, 45545584
"""

import keras
import os
import tensorflow as tf
import tensorflow_addons as tfa
from keras.layers import Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate

def context_module(input, filter):
    
    con1 = tfa.layers.InstanceNormalization()(input)
    con1 = tf.keras.layers.Conv2D(filter, (3, 3), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding="same")(con1)
    con2 = tf.keras.layers.Dropout(0.3)(con1)
    con3 = tfa.layers.InstanceNormalization()(con2)
    con3 = tf.keras.layers.Conv2D(filter, (3, 3), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding="same")(con3)
    return con3
	
def localization_module(input, filter):
    con1 = tf.keras.layers.Conv2D(filter, (3,3), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(input)
    con2 = tf.keras.layers.Conv2D(filter, (1,1), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(con1)
    
    return con2
	
def upsampling_module(input, filter):
    con1 = tf.keras.layers.UpSampling2D()(input)
    con1 = tf.keras.layers.Conv2D(filter, (3,3), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same')(con1)
    
    return con1
	
def improved_unet(height, width, channel):
    inputs = tf.keras.layers.Input(shape = (height, width, channel))
    
    con1 = tf.keras.layers.Conv2D(16, (3, 3), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same') (inputs)
    context_module1 = context_module(con1, 16)
    add1 = tf.keras.layers.Add()([con1, context_module1])
    
    con2 = tf.keras.layers.Conv2D(32, (3, 3),strides = 2, activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same') (add1)
    context_module2 = context_module(con2, 32)
    add2 = tf.keras.layers.Add()([con2, context_module2])
    
    con3 = tf.keras.layers.Conv2D(64, (3, 3),strides = 2, activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same') (add2)
    context_module3 = context_module(con3, 64)
    add3 = tf.keras.layers.Add()([con3, context_module3])
    
    con4 = tf.keras.layers.Conv2D(128, (3, 3),strides = 2, activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same') (add3)
    context_module4 = context_module(con4, 128)
    add4 = tf.keras.layers.Add()([con4, context_module4])
    
    con5 = tf.keras.layers.Conv2D(256, (3, 3),strides = 2, activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same') (add4)
    context_module5 = context_module(con5, 256)
    add5 = tf.keras.layers.Add()([con5, context_module5])
    
    upsampling_module1 = upsampling_module(add5, 128)
    c1 = tf.keras.layers.concatenate([upsampling_module1, add4])
    
    localization_module1 = localization_module(c1, 128)
    upsampling_module2 = upsampling_module(localization_module1, 64)
    c2 = tf.keras.layers.concatenate([upsampling_module2, add3])
    
    localization_module2 = localization_module(c2, 64)
    upsampling_module3 = upsampling_module(localization_module2, 32)
    c3 = tf.keras.layers.concatenate([upsampling_module3, add2])
    
    localization_module3 = localization_module(c3, 32)
    upsampling_module4 = upsampling_module(localization_module3, 16)
    c4 = tf.keras.layers.concatenate([upsampling_module4, add1])
    
    con6 = tf.keras.layers.Conv2D(32, (3, 3), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same') (c4)
    
    con7 = tf.keras.layers.Conv2D(1, (1, 1), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same') (localization_module2)
    u1 = tf.keras.layers.UpSampling2D()(con7)
    con8 = tf.keras.layers.Conv2D(1, (1, 1), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same') (localization_module3)
    add6 = tf.keras.layers.Add()([u1, con8])
    u2 = tf.keras.layers.UpSampling2D()(add6)
    con9 = tf.keras.layers.Conv2D(1, (1, 1), activation = tf.keras.layers.LeakyReLU(alpha=0.01), padding='same') (con6)
    add7 = tf.keras.layers.Add()([u2, con9])
    
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid', padding="same")(add7)
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
	
    return model


