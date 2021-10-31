"""
Originally created by me in Google Colab.

This file defines a UNet based on the architecture in https://arxiv.org/abs/1802.10508v1.
Note that this UNet is 3 layers deep as compared to the 5 layers in the paper.
"""

# Start by importing dependencies.

from tensorflow import keras
from tensorflow.keras import layers


# This is a dummy model, for bug-fixing the IO-pipeline.
def get_model(img_size_params=(128, 128, 3), num_classes=1):
    """
    Returns a basic convolution model.
    """
    inputs = keras.Input(shape=img_size_params)
    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.UpSampling2D(size=(2, 2))(x)
    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


# Only 3 layers deep for now.
def get_UNET_model(img_size_params=(128, 128, 3), num_classes=1):
    """
    Returns a UNet model as in https://arxiv.org/abs/1802.10508v1.

    """
    # Entry block
    inputs = keras.Input(shape=img_size_params)

    # First half of the network, encoder to downsample input and learn features

    conv1 = layers.Conv2D(16, 3, strides=1, padding="same")(inputs)
    # Split the above input. Then plug it into the below and across the skip connection.
    context1 = context_module(conv1, n_filters=16)
    add1 = layers.Add()([conv1, context1])

    skip1 = add1

    conv2 = layers.Conv2D(32, 3, strides=2, padding="same")(add1)  # + the split connection from prev conv3d layer
    context2 = context_module(conv2, n_filters=32)
    add2 = layers.Add()([conv2, context2])

    skip2 = add2

    conv3 = layers.Conv2D(64, 3, strides=2, padding="same")(add2)
    context3 = context_module(conv3, n_filters=64)
    add3 = layers.Add()([conv3, context3])

    # Second half of the network, the decoder.

    upsample1 = upsampling_module(add3, n_filters=32)

    concat1 = layers.Concatenate(axis=-1)([upsample1, skip2])
    localize1 = localization_module(concat1, n_filters=32)

    upsample2 = upsampling_module(localize1, n_filters=16)

    concat2 = layers.Concatenate(axis=-1)([upsample2, skip1])

    conv4 = layers.Conv2D(32, 3, strides=1, padding="same")(concat2)

    outputs = layers.Conv2D(num_classes, 1, activation="sigmoid", padding="same")(conv4)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model


"""
Modules used in the model.   
"""


def context_module(inputs, n_filters=32, drop_prob=0.3):
    """
    The context module consists of 2 convolution layers with a dropout layer in between.
    This is used to learn features as we go down the UNet.
    """
    conv1 = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), strides=1, padding="same")(inputs)
    leaky_relu1 = layers.LeakyReLU(0.01)(conv1)
    batch_norm1 = layers.BatchNormalization()(leaky_relu1)

    dropout = layers.Dropout(drop_prob)(batch_norm1)

    conv2 = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")(dropout)
    leaky_relu2 = layers.LeakyReLU(0.01)(conv2)
    batch_norm2 = layers.BatchNormalization()(leaky_relu2)

    return batch_norm2


def upsampling_module(inputs, n_filters=32):
    """
    The upsampling module upsamples low resolution feature maps while going up the UNet.
    """
    upsampling1 = layers.UpSampling2D(size=(2, 2))(inputs)
    conv1 = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")(upsampling1)
    leaky_relu1 = layers.LeakyReLU(0.01)(conv1)
    batch_norm1 = layers.BatchNormalization()(leaky_relu1)

    return batch_norm1


def localization_module(inputs, n_filters=32):
    """
    The localization module uses information from the lower layer on the UNet and skip connections to recreate
    the image masks.
    """
    conv1 = layers.Conv2D(filters=n_filters, kernel_size=(3, 3), padding="same")(inputs)
    leaky_relu1 = layers.LeakyReLU(0.01)(conv1)
    batch_norm1 = layers.BatchNormalization()(leaky_relu1)

    conv2 = layers.Conv2D(filters=n_filters/2, kernel_size=(1, 1), padding="same")(batch_norm1)
    leaky_relu2 = layers.LeakyReLU(0.01)(conv2)
    batch_norm2 = layers.BatchNormalization()(leaky_relu2)

    return batch_norm2


