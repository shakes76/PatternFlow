import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model, load_model

img_h = 4288
img_w = 2848
b_size = 32

def conv_block(tensor, nfilters, size=3, padding='same', initializer="he_normal"):
    block = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(tensor)
    block = BatchNormalization()(block)
    block = Activation("relu")(block)
    block = Conv2D(filters=nfilters, kernel_size=(size, size), padding=padding, kernel_initializer=initializer)(block)
    block = BatchNormalization()(block)
    block = Activation("relu")(block)
    return block


def deconv_block(tensor, residual, nfilters, size=3, padding='same', strides=(2, 2)):
    block = Conv2DTranspose(nfilters, kernel_size=(size, size), strides=strides, padding=padding)(tensor)
    block = concatenate([block, residual], axis=3)
    block = conv_block(block, nfilters)
    return block


