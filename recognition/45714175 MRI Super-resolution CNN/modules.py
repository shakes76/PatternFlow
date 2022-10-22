"""
modules.py
Contains function to return the model of the super-resolution CNN as well as the training 
callback function.
"""

from turtle import width
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from predict import predict
from constants import *

def get_model():
    """Returns super resolution CNN model"""
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    input = tf.keras.layers.Input(shape=(HEIGHT // DOWNSCALE_FACTOR, WIDTH // DOWNSCALE_FACTOR, 1))
    x = tf.keras.layers.Conv2D(64, 5, **conv_args)(input)
    x = tf.keras.layers.Conv2D(128, 3, **conv_args)(x)
    x = tf.keras.layers.Conv2D(128, 3, **conv_args)(x)
    x = tf.keras.layers.Conv2D(1 * (DOWNSCALE_FACTOR ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, DOWNSCALE_FACTOR)

    return tf.keras.Model(input, outputs)

class ESPCNCallback(tf.keras.callbacks.Callback):
    def __init__(self, testImage):
        super(ESPCNCallback, self).__init__()
        # downsample image
        self.testImage = tf.image.resize(testImage, 
        size=(HEIGHT // DOWNSCALE_FACTOR, WIDTH // DOWNSCALE_FACTOR), method=RESIZE_METHOD)

    def on_epoch_begin(self, epoch, logs=None):
        """Initialise PSNR array"""
        self.psnr = []

    def on_epoch_end(self, epoch, logs=None):
        """Compute and display PSNR metric"""
        psnrMean = np.mean(self.psnr)
        print("Mean PSNR for epoch: %.2f" % (psnrMean))
        predict(self.model, self.testImage)

    def on_test_batch_end(self, batch, logs=None):
        """Append psnr metrics to array"""
        self.psnr.append(10 * math.log10(1 / logs["loss"]))

    def on_train_end(self, logs=None):
        """Display psnr over epochs"""
        plt.figure(figsize=(15, 10))
        plt.plot(self.psnr)
        plt.title("PSNR per epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Mean PSNR")
        plt.show()