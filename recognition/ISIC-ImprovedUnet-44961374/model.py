"""
This script creates an Improved U-Net model.
Specifications of the model can be found at: https://arxiv.org/abs/1802.10508v1
@author: Mujibul Islam Dipto
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose, LeakyReLU
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import  Conv2D


# GLOBAL CONSTANTS (values taken from the paper)
P_DROP = 0.3 # dropout probability
NUM_STRIDES = 2 # number of input strides
LEAKY_RELU_ALPHA = 1e-2 # slope of LeakyReLU
IMAGE_ROWS = 256 # image x dimension
IMAGE_COLS = 256 # image y dimension
IMAGE_CHANNELS = 1 # greyscale
KERNEL_SIZE = (3,3) # size of kernel


def create_conv2d(input_layer, filters):
    """Creates a Conv2D layer based on the Improved UNet architecture. 
    Please look at the README.md for more details.

    Args:
        input_layer (keras.layer): input layer to this conv2d layer
        filters (int): number of filters

    Returns:
        keras.layer.Conv2D: the conv2d layer that has been created
    """
    conv2d_layer = Conv2D(filters=filters, kernel_size=KERNEL_SIZE, padding='same', activation=LeakyReLU(alpha=LEAKY_RELU_ALPHA))(input_layer)
    return conv2d_layer


def context_module(input_layer, filters):
    instance_norm_layer1 = InstanceNormalization()(input_layer)
    conv_layer1 = create_conv2d(instance_norm_layer1, filters)
    dropout = Dropout(P_DROP)(conv_layer1)
    instance_norm_layer2 = InstanceNormalization()(dropout)
    conv_layer2 = create_conv2d(instance_norm_layer2, filters)
    return conv_layer2

def upsampling_module():
    pass

def localization_module():
    pass

def create_model():
    pass

