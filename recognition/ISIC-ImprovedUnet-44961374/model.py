"""
This script creates an Improved U-Net model.
Specifications of the model can be found at: https://arxiv.org/abs/1802.10508v1
@author: Mujibul Islam Dipto
"""
from tensorflow.keras.layers import Conv2D, Dropout, Input, concatenate, Add, UpSampling2D, LeakyReLU
from tensorflow.keras import Model
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.keras.layers import Conv2D


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
    conv2d_layer = Conv2D(filters=filters, kernel_size=kernel_size, 
        strides=strides, padding='same', activation=LeakyReLU(alpha=LEAKY_RELU_ALPHA))(input_layer)
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
    upsampling_layer = UpSampling2D((2,2))(input_layer)
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
    conv_layer_2 = create_conv2d(conv_layer_1, filters, (1, 1), INIT_STRIDES)
    return conv_layer_2


def segmentation_layer(input_layer, filters):
    """Creates a segmentation layer based on the Improved UNet architecture.
    Contains a Conv2D layer with the default parameters of this model

    Args:
        input_layer (keras.layer): input layer to this layer
        filters (int): number of filters

    Returns:
        keras.layer: output conv2d layer
    """
    conv2d = create_conv2d(input_layer, filters, KERNEL_SIZE, INIT_STRIDES)
    return conv2d


def create_model(output_channels):
    """Creates an Improved UNet model based on the paper's specifications.
    The inline comments refer directly to the annotated image of the model which
    can be found in the README.md
    """
    ########## INPUT ##########
    input_layer = Input(shape=(IMAGE_ROWS, IMAGE_COLS, IMAGE_CHANNELS))
    
    ########## CONTRACTING PATH ##########
    # level 1
    conv_layer_1 = create_conv2d(input_layer, INIT_NO_FILTERS, KERNEL_SIZE, INIT_STRIDES) # 3x3 conv
    context_1 = context_module(conv_layer_1, INIT_NO_FILTERS) # context module
    add_layer_1= Add()([conv_layer_1, context_1]) # element-wise sum

    # level 2
    conv_layer_2 = create_conv2d(add_layer_1, INIT_NO_FILTERS * 2, KERNEL_SIZE, (2, 2)) # 3x3 stride 2 conv
    context_2 = context_module(conv_layer_2, INIT_NO_FILTERS * 2) # context module
    add_layer_2 = Add()([conv_layer_2, context_2]) # element-wise sum

    # level 3
    conv_layer_3 = create_conv2d(add_layer_2, INIT_NO_FILTERS * 4, KERNEL_SIZE, (2, 2))  # 3x3 stride 2 conv
    context_3 = context_module(conv_layer_3, INIT_NO_FILTERS * 4) # context module
    add_layer_3 = Add()([conv_layer_3, context_3]) # element-wise sum

    # level 4
    conv_layer_4 = create_conv2d(add_layer_3, INIT_NO_FILTERS * 8, KERNEL_SIZE, (2, 2))  # 3x3 stride 2 conv
    context_4 = context_module(conv_layer_4, INIT_NO_FILTERS * 8) # context module
    add_layer_4 = Add()([conv_layer_4, context_4]) # element-wise sum

    # base
    conv_layer_5 = create_conv2d(add_layer_4, INIT_NO_FILTERS * 16, KERNEL_SIZE, (2, 2))  # 3x3 stride 2 conv
    context_5 = context_module(conv_layer_5, INIT_NO_FILTERS * 16) # context module
    add_layer_5 = Add()([conv_layer_5, context_5]) # element-wise sum

    ########## EXPANSIVE PATH ##########
    # base
    upsample_1 = upsampling_module(add_layer_5, INIT_NO_FILTERS * 8) # upsampling module

    # level 4
    concat_1 = concatenate([upsample_1, add_layer_4]) # concatenation
    localization_1 = localization_module(concat_1, INIT_NO_FILTERS * 8) # localization module
    up_sample_2 = upsampling_module(localization_1, INIT_NO_FILTERS * 4) # upsampling module

    # level 3
    concat_2 = concatenate([up_sample_2, add_layer_3]) # concatenation
    localization_2 = localization_module(concat_2, INIT_NO_FILTERS * 4) # localization module
    segmentation_1 = segmentation_layer(localization_2, INIT_NO_FILTERS) # segmentation layer
    segmentation_1_up = UpSampling2D()(segmentation_1) # upscale
    up_sample_3 = upsampling_module(localization_2, INIT_NO_FILTERS * 2) # upsampling module


    # level 2
    concat_3 = concatenate([up_sample_3, add_layer_2]) # concatenation
    localization_3 = localization_module(concat_3, INIT_NO_FILTERS * 2) # localization module
    segmentation_2 = segmentation_layer(localization_3, INIT_NO_FILTERS) # segmentation layer
    segmentation_2 = Add()([segmentation_2, segmentation_1_up])  # element-wise sum
    segmentation_2 = UpSampling2D()(segmentation_2)  # upsacle
    up_sample_4 = upsampling_module(localization_3, INIT_NO_FILTERS) # upsampling module

    # level 1
    concat_4 = concatenate([up_sample_4, add_layer_1]) # concatenation
    conv_layer_6 = create_conv2d(concat_4, INIT_NO_FILTERS * 2, KERNEL_SIZE, INIT_STRIDES) # 3x3 conv
    segmentation_3 = segmentation_layer(conv_layer_6, INIT_NO_FILTERS) # segmentation layer
    segmentation_3 = Add()([segmentation_3, segmentation_2]) # element-wise sum

    # output layers
    output = Conv2D(output_channels, KERNEL_SIZE, activation="softmax", padding="same")(segmentation_3) # softmax
    model = Model(name="ImprovedUnet", inputs=input_layer, outputs=output) # final model
    return model
