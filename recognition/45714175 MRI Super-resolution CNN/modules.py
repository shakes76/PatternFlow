"""
modules.py
Contains function to return the model of the super-resolution CNN
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras import Model

def get_model():
    """Returns super resolution CNN model"""
    width, height = 256, 240
    upscale_factor = 4
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    input = tf.keras.layers.Input(shape=(width, height, 1))
    x = tf.keras.layers.Conv2D(64, 5, **conv_args)(input)
    x = tf.keras.layers.Conv2D(128, 3, **conv_args)(x)
    x = tf.keras.layers.Conv2D(128, 3, **conv_args)(x)
    x = tf.keras.layers.Conv2D(1 * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return tf.keras.Model(input, outputs)