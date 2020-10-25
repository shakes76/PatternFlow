"""
COMP3710 Report 

@author Huizhen 
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose


print("TensorFlow Version: ", tf.__version__) 

# Define Block Functions
def context_module(inputs, filters, dropout_rate):
    bn1 = BatchNormalization()(inputs)
    relu1 = Activation('relu')(bn1)
    conv1 = Conv2D(filters, (3,3), padding='same')(relu1)
    dropout = Dropout(dropout_rate)(conv1)
    bn2 = BatchNormalization()(dropout)
    relu2 = Activation('relu')(bn2)
    conv2 = Conv2D(filters, (3,3), padding='same')(relu2)
    output = Add()([inputs, conv2])
    return output

def localization_module(upsample, cm, filters):
    concat = concatenate([upsample, cm])
    conv1 = Conv2D(filters*2, (3,3), padding='same')(concat)
    conv2 = Conv2D(filters, (1,1))(conv1)
    return conv2

def upsampling_module(inputs, filters):
    up = UpSampling2D(size=(2,2))(inputs)
    conv = Conv2D(filters, (3,3), padding='same')(up)
    return conv

def segmentation_addup_module(inputs1, inputs2):
    seg1 = Conv2DTranspose(1, (2,2), strides=(2,2), padding='same')(inputs1)
    seg2 = Conv2DTranspose(1, (2,2), strides=(1,1), padding='same')(inputs2)
    addup = Add()([seg1, seg2])
    return addup


def improved_unet():
    
    inputs = Input((128,128,3))

    c1 = Conv2D(16, (3,3), padding='same')(inputs)
    cm1 = context_module(c1, 16, 0.3)
    
    c2 = Conv2D(32, (3,3), strides=2, padding='same')(cm1)
    cm2 = context_module(c2, 32, 0.3)
    
    c3 = Conv2D(64, (3,3), strides=2, padding='same')(cm2)
    cm3 = context_module(c3, 64, 0.3)
    
    c4 = Conv2D(128, (3,3), strides=2, padding='same')(cm3)
    cm4 = context_module(c4, 128, 0.3)
    
    c5 = Conv2D(256, (3,3), strides=2, padding='same')(cm4)
    cm5 = context_module(c5, 256, 0.3)
    
    u1 = upsampling_module(cm5, 128)
    
    local1 = localization_module(u1, cm4, 128)
    
    u2 = upsampling_module(local1, 64)
    
    local2 = localization_module(u2, cm3,  64)
        
    u3 = upsampling_module(local2, 32)
    
    local3 = localization_module(u3, cm2,  32)
    
    u4 = upsampling_module(local3, 16)
    
    concat = concatenate([u4, cm1])
    
    c6 = Conv2D(32, (3,3), padding='same')(concat)
        
    sum1 = segmentation_addup_module(local2, local3)
    sum2 = segmentation_addup_module(sum1, c6)
    
    outputs = Activation('sigmoid')(sum2)
    
    network = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    return network



model = improved_unet()
model.summary()
# parameters

# layers

# build networks

# build model

# trian

# test

# plot