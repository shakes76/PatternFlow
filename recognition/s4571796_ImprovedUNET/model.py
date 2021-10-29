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

def segmentation_module(input, output_filters):
    """Performs the convolution in the localization module

    Arguments:
    input: The image to perform convolution on
    output_filters: The number of classes of the output image
    """
    # Used 1*1 filter size
    x = Conv2D(output_filters, 1, padding="same", activation=LeakyReLU(alpha=0.1))(input)

    return x