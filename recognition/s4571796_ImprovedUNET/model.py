#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Joshua Yu Xuan Soo"
__studentID__ = "s4571796"
__email__ = "s4571796@uqconnect.edu.au"

"""
Improved UNET Model for Image Segmentation on ISIC Melanoma Dataset.

References: 
    https://arxiv.org/pdf/1802.10508v1.pdf

"""

# Imports
from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, Dropout, Concatenate, LeakyReLU, UpSampling2D, BatchNormalization
import tensorflow as tf

# Functions

def context_module(input, num_filters):
    """Performs the convolution in the context module

    Arguments:
    input: The image to perform convolution on
    num_filters: The number of filters used to perform convolution
    """
    # Store the initial values of the image
    temp = input

    # Used 3*3 filter size
    # Activation Function is LeakyReLU
    x = Conv2D(num_filters, 3, padding="same", activation=LeakyReLU(alpha=0.1))(input)

    # BatchNormalization to reduce Overfitting
    x = BatchNormalization()(x)

    # Dropout to reduce Overfitting, value = 0.3
    x = Dropout(0.1)(x)

    # UNET Architecture performs Convolution2D two times for a single layer
    x = Conv2D(num_filters, 3, padding="same", activation=LeakyReLU(alpha=0.1))(x)

    # Add the initial values to the convoluted result element wise
    y = tf.math.add(temp, x)

    return y

def downs(input, num_filters):
    """Performs the full down convolution per layer of UNET. To half dimensions
     of image, convolution with stride 2 is included.

    Arguments:
    input: The image to perform convolution on
    num_filters: The number of filters used to perform convolution
    """
    x = context_module(input, num_filters)

    # Choosing Filter Size of 3*3, Stride set to Filter Size = 2*2. Note that
    # the number of filters of this convolution is doubled.
    p = Conv2D((num_filters * 2), 3, padding="same", strides=(2,2), activation=LeakyReLU(alpha=0.1))(x)
    return x, p

def localization_module(input, num_filters):
    """Performs the convolution in the localization module

    Arguments:
    input: The image to perform convolution on
    num_filters: The number of filters used to perform convolution
    """
    # Used 3*3 filter size
    # Activation Function is LeakyReLU
    x = Conv2D(num_filters, 3, padding="same", activation=LeakyReLU(alpha=0.1))(input)

    # UNET Architecture performs Convolution2D two times for a single layer
    x = Conv2D(num_filters, 1, padding="same", activation=LeakyReLU(alpha=0.1))(x)

    return x

def ups(input, skip_features, num_filters):
    """Performs the full up convolution per layer of UNET. Similar to downs, 
    uses Conv2dTranspose to double dimensions of image and Concatenates 
    current image with skip connections, then performs a localization module.

    Arguments:
    input: The image to perform convolution on
    skip_features: Feature to skip, pass in input from downsampling
    num_filters: The number of filters used to perform convolution
    """
    x = Conv2DTranspose(num_filters, 3, strides=2, padding="same", activation=LeakyReLU(alpha=0.1))(input)
    x = Concatenate()([x, skip_features])
    x = localization_module(x, num_filters)

    return x

def ups_end(input, skip_features, num_filters):
    """A clone of the ups function, but instead of performing the localization
    module at the end, only perform a single convolution with kernel size 3.

    Arguments:
    input: The image to perform convolution on
    skip_features: Feature to skip, pass in input from downsampling
    num_filters: The number of filters used to perform convolution
    """

    x = Conv2DTranspose(num_filters, 3, strides=2, padding="same", activation=LeakyReLU(alpha=0.1))(input)
    x = Concatenate()([x, skip_features])
    x = Conv2D((num_filters * 2), 3, padding="same", activation=LeakyReLU(alpha=0.1))(x)

    return x

def segmentation_module(input, output_filters):
    """Performs the convolution in the localization module

    Arguments:
    input: The image to perform convolution on
    output_filters: The number of classes of the output image
    """
    # Used 1*1 filter size
    x = Conv2D(output_filters, 1, padding="same", activation=LeakyReLU(alpha=0.1))(input)

    return x

def unet_model(num_channels, image_height, image_width, image_channels):
    """Builds the UNET Model using defined variable values

    Arguments:
    num_channels: The number of channels for the output image, i.e. 1 for Grayscale, 3 for RGB, etc.
    image_height: The height of the original image
    image_width: The width of the original image
    image_channels: The number of channels on the original image

    Notes:
        This project aims to classify Melanoma. The number of channels for the output is 2 because the
        segmentations contain the background as black, and the melanoma as white. The original number 
        of channels is 3, since it is an RGB Image.
    """
    inputs = Input((image_height, image_width, image_channels))

    layer0 = Conv2D(16, 3, padding="same", activation=LeakyReLU(alpha=0.1))(inputs)

    # Inputting the original image and iterative downsampling
    # Store Layers to transfer as skip connections to upsampling step
    # Store Poolings to pass to next layers as inputs
    layer1, pool1 = downs(layer0, 16)
    layer2, pool2 = downs(pool1, 32)
    layer3, pool3 = downs(pool2, 64)
    layer4, pool4 = downs(pool3, 128)

    # Bridge: Only context module applied here, filters = 256
    bridge = context_module(pool4, 256)

    # Inputting the bridge and iterative upsampling
    # Taking in the layers defined earlier as skip connections
    # First upsampling does not need segmentation layer
    layer5 = ups(bridge, layer4, 128)
    layer6 = ups(layer5, layer3, 64)
    layer7 = ups(layer6, layer2, 32)
    layer8 = ups_end(layer7, layer1, 16)
    
    # First upsampling does not need segmentation layer, but layer6 
    # onwards require segmentation layer
    seg1 = segmentation_module(layer6, 2)
    # Upsampling first segmentation layer
    seg1 = UpSampling2D(size=(2,2), interpolation='nearest')(seg1)
    seg2 = segmentation_module(layer7, 2)
    # First element wise addition
    seg_add1 = tf.math.add(seg1, seg2)
    # Upsampling the sum of the first and second segmentation layer
    seg_add1 = UpSampling2D(size=(2,2), interpolation='nearest')(seg1)
    seg3 = segmentation_module(layer8, 2)
    # Second element wise addition
    segmentation = tf.math.add(seg_add1, seg3)

    # Utilizing Softmax Function
    outputs = Conv2D(num_channels, (1,1), activation="softmax")(segmentation)
    model = Model(inputs, outputs)
    return model