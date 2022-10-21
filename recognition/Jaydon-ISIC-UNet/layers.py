'''
    layers.py
    Author: Jaydon Hansen
    Date created: 26/10/2020
    Date last modified: 7/11/2020
    Python Version: 3.8
'''

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Concatenate,
    BatchNormalization,
)
from tensorflow.keras.initializers import HeNormal


def downsample(
    input,
    filters,
    kernel_size=(3, 3),
    padding="same",
    strides=(1, 1),
    kernel_initializer=HeNormal(),
):
    """
    Generates a downsampling block. 2 x 3x3 convolutional layers -> 2x2 max pooling
    """
    # Convolutional layer 1
    conv = Conv2D(
        filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation="relu",
        kernel_initializer=kernel_initializer,
    )(input)
    norm = BatchNormalization()(conv)

    # Convolutional layer 2
    conv = Conv2D(
        filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation="relu",
        kernel_initializer=kernel_initializer,
    )(norm)
    norm = BatchNormalization()(conv)

    # Pooling layer
    pooling = MaxPooling2D((2, 2), (2, 2))(norm)
    return norm, pooling


def upsample(
    input,
    skip,
    filters,
    kernel_size=(3, 3),
    padding="same",
    strides=(1, 1),
    kernel_initializer=HeNormal(),
):
    """
    Generates an upsampling block. Deconv with stride 2 -> concatenate -> 2x 3x3 convolutional layers
    """
    # Upsampling and concatenation
    upsample = UpSampling2D()(input)  # deconvolution layer
    concatenate = Concatenate()(
        [upsample, skip]
    )  # concatenate the skip and the deconvoluted layer together

    # Convolutional layer 1
    conv = Conv2D(
        filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation="relu",
        kernel_initializer=kernel_initializer,
    )(concatenate)
    norm = BatchNormalization()(conv)

    # Convolutional layer 2
    conv = Conv2D(
        filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation="relu",
        kernel_initializer=kernel_initializer,
    )(norm)
    norm = BatchNormalization()(conv)
    return norm


def bottleneck(
    input,
    filters,
    kernel_size=(3, 3),
    padding="same",
    strides=(1, 1),
    kernel_initializer=HeNormal(),
):
    """
    Bottleneck to connect the downsampling and upsampling blocks
    """
    # Convolutional layer 1
    conv = Conv2D(
        filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation="relu",
        kernel_initializer=kernel_initializer,
    )(input)
    norm = BatchNormalization()(conv)  # normalize between layers
    # Convolutional layer 2
    conv = Conv2D(
        filters,
        kernel_size=kernel_size,
        padding=padding,
        strides=strides,
        activation="relu",
        kernel_initializer=kernel_initializer,
    )(norm)
    norm = BatchNormalization()(conv)  # normalize between layers

    return norm
