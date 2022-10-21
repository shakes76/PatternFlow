from pkgutil import get_data
import numpy as np
import tensorflow as tf
import tensorflow.keras.utils
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import PIL

import dataset


def get_model(upscale_factor=4, channels=1):
    conv_args= {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }


    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)
