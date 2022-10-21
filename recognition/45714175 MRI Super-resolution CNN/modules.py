"""
modules.py
Contains function to return the model of the super-resolution CNN
"""

from turtle import width
import tensorflow as tf
import numpy as np
import math
from predict import predict

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

class ESPCNCallback(tf.keras.callbacks.Callback):
    def __init__(self, testImage):
        super(ESPCNCallback, self).__init__()
        # downsample image
        self.testImage = testImage.resize((240 // 4, 256 // 4), method="gaussian")

    def on_epoch_begin(self, epoch, logs=None):
        """Initialise PSNR array"""
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        """Compute and display PSNR metric"""
        psnrMean = np.mean(self.psnr)
        print("Mean PSNR for epoch: %.2f" % (psnrMean))
        predict(self.model, self.testImage, epoch)

    def on_test_batch_end(self, batch, logs=None):
        """Append psnr metrics to array"""
        self.psnr.append(10 * math.log10(1 / logs["loss"]))