import tensorflow as tf
import tnsorflow_addons as tfa
from tensorflow.keras import layers, models

#method for created the improved unet model
def improved_unet():
    return model

#methods for different modules of the unet


def conv_relu_block(input, conv_depth, conv_size=(3,3)):
    #convolutions with leaky relu, and normalisation
    conv = layers.Conv2D(conv_depth, kernel_size=conv_size, padding='same')(input)
    conv = layers.batchNormalisation()(conv)
    conv = layers.LeakyReLu(alpha=0.01)(conv)
    return conv

def context_module(input, conv_depth):
    # context module contains 2 convolutions with a dropout of 0.3 in between
    # with added element-wise sum
    cont = conv_relu_block(input, conv_depth)
    cont = layers.Dropout(0.3)(cont)
    cont = conv_relu_block(cont,conv_depth)
    cont = layers.Add()([input, cont])
    return cont

def localization_module(input, conv_depth):
    # a localization module contains one 3x3 convolution
    # and a 1x1 convolution that halves the number of features
    loc = conv_relu_block(input, conv_depth)
    loc = layers.Conv2D(conv_depth, kernel_size=(1, 1), padding='same')(loc)
    return loc

def upsampling_module(input, conv_depth):
    up = layers.UpSampling2D((2,2))(input)
    up = layers.Conv2D(conv_depth, kernel_size=(3, 3), padding='same')(up)
    return up
def segmentation_layer(input):
    seg = layers.Conv2D(1, (1,1),activation = 'softmax')(input)
    return seg

#method for blocks in the encoder path
def encoder_block():
    return en
#method for blocks in the decoder path
def decoder_block():
    return de






