"""modules.py

The source code for components of the super-resolution model. 
"""

import math
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras import layers

from constants import DOWNSAMPLE_FACTOR, IMG_DOWN_WIDTH, IMG_DOWN_HEIGHT
from predict import display_prediction

def get_model(
    upscale_factor: int = DOWNSAMPLE_FACTOR,
    img_width: int = IMG_DOWN_WIDTH,
    img_height: int = IMG_DOWN_HEIGHT,
    channels: int = 1,
) -> keras.Model:
    """Return a super-resolution model

    Args:
        upscale_factor (int, optional): Multiple to upscale the final output. 
            Defaults to 4.
        img_width (int, optional): Input image width. Defaults to 64.
        img_height (int, optional): Input image height. Defaults to 60.
        channels (int, optional): Number of colour channels. Defaults to 1.

    Returns:
        keras.Model: Super-resolution model.
    """
    # https: // keras.io / examples / vision / super_resolution_sub_pixel /
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }

    inputs = keras.Input(shape=(img_width, img_height, channels))
    next_layer = layers.Conv2D(64, 5, **conv_args)(inputs)
    next_layer = layers.Conv2D(128, 3, **conv_args)(next_layer)
    next_layer = layers.Conv2D(128, 3, **conv_args)(next_layer)
    next_layer = layers.Conv2D(
        channels * (upscale_factor ** 2),
        3,
        **conv_args
    )(next_layer)

    outputs = tf.nn.depth_to_space(next_layer, upscale_factor)

    return keras.Model(inputs, outputs)


class ESPCNCallback(keras.callbacks.Callback):
    """Callback methods for the ESPCN model"""

    def __init__(self, test_image):
        """Initialise callback class and attributes

        Args:
            test_image: a full-res image to be used for outputting prediction
                results.
        """
        super(ESPCNCallback, self).__init__()
        self.test_image = test_image
        self.psnr = np.array([])
        self.psnrs = []  # Record of each epoch's psnr value

    def on_epoch_begin(self, epoch, logs=None):
        """On epoch begin, initialise array to store psnr values."""
        self.psnr = np.array([])

    def on_epoch_end(self, epoch, logs=None):
        """On epoch end, record PSNR value"""
        psnr = np.mean(self.psnr)
        self.psnrs.append(psnr)
        print(f"\nMean PSNR for epoch: {psnr:2f}")
        if epoch % 5 == 0:
            display_prediction(self.test_image, self.model, f"Epoch {epoch}")

    def on_test_batch_end(self, batch, logs=None):
        """On batch end, append the next psnr value"""
        self.psnr = np.append(self.psnr, 10 * math.log10(1 / logs["loss"]))

    def on_train_end(self, logs=None):
        """On end of training (.fit), display the psnr record"""
        plt.figure(figsize=(15, 10))
        plt.plot(self.psnrs)
        plt.title("PSNR per epoch")
        plt.xlabel("Epoch")
        plt.ylabel("mean PSNR")
        plt.show()
