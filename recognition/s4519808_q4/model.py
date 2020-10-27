"""
COMP3710 Report 

This file contains functions to build the imporved unet.

@author Huizhen 
"""

import tensorflow as tf
tf.random.Generator = None
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose, LeakyReLU

print("TensorFlow Version: ", tf.__version__) 


# Define Block Functions
def context_module(inputs, filters):
    """ filters is the output size of the module"""
    bn1 = tfa.layers.InstanceNormalization()(inputs)
    relu1 = LeakyReLU(alpha=0.01)(bn1)
    conv1 = Conv2D(filters, (3,3), padding='same')(relu1)
    dropout = Dropout(0.3)(conv1)
    bn2 = tfa.layers.InstanceNormalization()(dropout)
    relu2 = LeakyReLU(alpha=0.01)(bn2)
    conv2 = Conv2D(filters, (3,3), padding='same')(relu2)
    #output = Add()([inputs, conv2])
    return conv2

def upsampling_module(inputs, filters):
    """ filters is the output size of the module"""
    up = UpSampling2D(size=(2,2))(inputs)
    conv = Conv2D(filters, (3,3), padding='same')(up)
    return conv

def localization_module(inputs, filters):
    """ filters is the output size of the module"""
    #concat = concatenate([upsample, cm])
    conv1 = Conv2D(filters*2, (3,3), padding='same')(inputs)
    conv2 = Conv2D(filters, (1,1))(conv1)
    return conv2


# Build the Model
def improved_unet(h, w):
    """cm, um, lm stand for differnet modules"""
    # input layer
    inputs = Input((h,w,3))
    
    conv1 = Conv2D(16, (3,3), padding='same')(inputs)
    cm1 = context_module(conv1, 16)
    add1 = Add()([conv1, cm1]) # concat later
    
    conv2_stride = Conv2D(32, (3,3), strides=2, padding='same')(add1)
    cm2 = context_module(conv2_stride, 32)
    add2 = Add()([conv2_stride, cm2]) # concat later
    
    conv3_stride = Conv2D(64, (3,3), strides=2, padding='same')(add2)
    cm3 = context_module(conv3_stride, 64)
    add3 = Add()([conv3_stride, cm3]) # concat later
    
    conv4_stride = Conv2D(128, (3,3), strides=2, padding='same')(add3)
    cm4 = context_module(conv4_stride, 128)
    add4 = Add()([conv4_stride, cm4]) # concat later
    
    conv5_stride = Conv2D(256, (3,3), strides=2, padding='same')(add4)
    cm5 = context_module(conv5_stride, 256)
    add5 = Add()([conv5_stride, cm5])
    
    um1 = upsampling_module(add5, 128)
    concat1 = concatenate([um1, add4])
    lm1 = localization_module(concat1, 128)
    
    um2 = upsampling_module(lm1, 64)
    concat2 = concatenate([um2, add3])
    lm2 = localization_module(concat2, 64) # addup later
    
    um3 = upsampling_module(lm2, 32)
    concat3 = concatenate([um3, add2])
    lm3 = localization_module(concat3, 32) # addup later
    
    um4 = upsampling_module(lm3, 16)
    concat4 = concatenate([um4, add1])
    conv6 = Conv2D(32, (3,3), padding='same')(concat4) # addup later
    
    seg1 = Conv2D(1, (1,1), padding='same')(lm2)
    seg1 = UpSampling2D(size=(2,2))(seg1)
    seg2 = Conv2D(1, (1,1), padding='same')(lm3)
    sum1 = Add()([seg1, seg2])
    sum1 = UpSampling2D(size=(2,2))(sum1)
    seg3 = Conv2D(1, (1,1), padding='same')(conv6)
    sum2 = Add()([sum1, seg3])
    
    outputs = Activation('sigmoid')(sum2)
    network = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    return network








