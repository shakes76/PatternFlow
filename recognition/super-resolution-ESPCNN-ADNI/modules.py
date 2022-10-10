"""model.py

Functions to retrieve the super-resolution model.
"""

import tensorflow as tf
from tensorflow import keras
from keras import layers


def get_model(upscale_factor=3, channels=1) -> keras.Model:
    # https: // keras.io / examples / vision / super_resolution_sub_pixel /
    conv_args = {
        "activation": "relu",
        "kernel_initialiser": "Orthogonal",
        "padding": "same",
    }

    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)
