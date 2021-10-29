"""
This script creates an Improved U-Net model.
Specifications of the model can be found at: https://arxiv.org/abs/1802.10508v1
@author: Mujibul Islam Dipto
"""
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose, LeakyReLU
from tensorflow.python.eager.context import context
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import  Conv2D


# GLOBAL CONSTANTS (values taken from the paper)
P_DROP = 0.3 # dropout probability
NUM_STRIDES = 2 # number of input strides
LEAKY_RELU_ALPHA = 1e-2 # slope of LeakyReLU
IMAGE_ROWS = 256 # image x dimension
IMAGE_COLS = 256 # image y dimension
IMAGE_CHANNELS = 1 # greyscale
KERNEL_SIZE = (3, 3) # size of kernel
INIT_NO_FILTERS = 16 # initial number of filters
INIT_STRIDES = (1, 1) # initial strides


def create_conv2d(input_layer, filters, kernel_size, strides):
    """Creates a Conv2D layer based on the Improved UNet architecture. 
    Please look at the README.md for more details.

    Args:
        input_layer (keras.layer): input layer to this conv2d layer
        filters (int): number of filters
        kernel_size (tuple): size of the kernel

    Returns:
        keras.layer.Conv2D: the conv2d layer that has been created
    """
    conv2d_layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', activation=LeakyReLU(alpha=LEAKY_RELU_ALPHA))(input_layer)
    return conv2d_layer


def context_module(input_layer, filters):
    """Creates a context module based on the Improved UNet architecture. Contains a pre-activation residual block with 
    two 3x3 conv layers and a dropout layer with probability = 0.3
    These modules are connected by 3x3 convs with input stride 2 to reduce the resolution of the feature maps.
    This allows for more features while descending down the aggregation pathway.
    Source: https://arxiv.org/pdf/1802.10508v1.pdf

    Args:
        input_layer (keras.layer): input layer to this module
        filters (int): number of filters

    Returns:
        [keras.layer]: final layer of this module
    """
    instance_norm_layer_1 = InstanceNormalization()(input_layer)
    conv_layer_1 = create_conv2d(instance_norm_layer_1, filters, KERNEL_SIZE, INIT_STRIDES)
    dropout = Dropout(P_DROP)(conv_layer_1)
    instance_norm_layer_2 = InstanceNormalization()(dropout)
    conv_layer_2 = create_conv2d(instance_norm_layer_2, filters, KERNEL_SIZE, INIT_STRIDES)
    return conv_layer_2

def upsampling_module(input_layer, filters):
    """Creates an upsampling module based on the Improved Unet architecture.
    Performs 2d upsampling, followed by 3x3 convolution.

    Args:
        input_layer (keras.layer): input layer to this module
        filters (int): number of filters

    Returns:
        keras.layer: final layer of this module
    """
    upsampling_layer = UpSampling2D(KERNEL_SIZE)(input_layer)
    conv2d = create_conv2d(upsampling_layer, filters, KERNEL_SIZE, INIT_STRIDES)
    return conv2d

def localization_module(input_layer, filters):
    """Creates a localization module based on the Improved UNet architecture.
    Contains two conv2d layers with a LeakyReLU activation function and slope = 0.01

    Args:
        input_layer (keras.layer): input layer to this module
        filters (int): number of filters

    Returns:
        keras.layer: final layer of this module
    """
    conv_layer_1 = create_conv2d(input_layer, filters, KERNEL_SIZE, INIT_STRIDES)
    conv_layer_2 = create_conv2d(conv_layer_1, filters, (1,1))
    return conv_layer_2

def create_model():
    ########## INPUT ##########
    input_layer = Input(shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))

    ########## CONTRACTING PATH ##########
    # level 1
    conv_layer_1 = create_conv2d(input_layer, INIT_NO_FILTERS, KERNEL_SIZE, INIT_STRIDES)
    context_1 = context_module(conv_layer_1, INIT_NO_FILTERS)
    add_layer_1= Add()([conv_layer_1, context_1])
    # level 2
    conv_layer_2 = create_conv2d(add_layer_1, INIT_NO_FILTERS * 2, KERNEL_SIZE, (2,2))
    context_2 = context_module(conv_layer_2, INIT_NO_FILTERS * 2)
    add_layer_2 = Add()([conv_layer_2, context_2]) 

create_model()

