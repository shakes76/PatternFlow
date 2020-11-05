"""
Segment the ISICs data set with the Improved UNet 
@author Nan WANG 
"""

import tensorflow as tf
tf.random.Generator = None
from tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose, LeakyReLU


# build context module
def context_module(input, filters):
    ins_layer1 = InstanceNormalization()(input)
    relu_layer1 = LeakyReLU(alpha=0.01)(ins_layer1)
    conv_layer1 = Conv2D(filters, (3,3), padding='same')(relu_layer1)
    dropout = Dropout(0.3)(conv_layer1)
    ins_layer2 = tfa.layers.InstanceNormalization()(dropout)
    relu_layer2 = LeakyReLU(alpha=0.01)(ins_layer2)
    conv_layer2 = Conv2D(filters, (3,3), padding='same')(relu_layer2)
    return conv_layer2

# build unsampling module
def upsampling_module(input, filters):
    upsample = UpSampling2D(size=(2,2))(input)
    conv_layer = Conv2D(filters, (3,3), padding='same')(upsample)
    return conv_layer

# build localization module
def localization_module(input, filters):
    conv_layer1 = Conv2D(filters*2, (3,3), padding='same')(input)
    conv_layer2 = Conv2D(filters, (1,1))(conv_layer1)
    return conv_layer2








