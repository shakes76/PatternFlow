"""
Segment the ISICs data set with the Improved UNet 
@author Nan WANG 
"""
import tensorflow as tf
tf.random.Generator = None
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose, LeakyReLU


# build context module
def context_module(input, filters):
    ins_layer1 = InstanceNormalization()(input)
    conv_layer1 = Conv2D(filters, (3,3), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(ins_layer1)
    dropout = Dropout(0.3)(conv_layer1)
    ins_layer2 = InstanceNormalization()(dropout)
    conv_layer2 = Conv2D(filters, (3,3), padding='same',activation=tf.keras.layers.LeakyReLU(alpha=0.01))(ins_layer2)
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








