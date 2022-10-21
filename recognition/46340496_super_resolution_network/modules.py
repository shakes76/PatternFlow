import tensorflow as tf
import matplotlib.pyplot as plt
from dataset import *

# building the model
def get_model(upscale_factor=4, channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = tf.keras.Input(shape=(None, None, channels))
    x = tf.keras.layers.Conv2D(64, 5, **conv_args)(inputs)
    x = tf.keras.layers.Conv2D(64, 3, **conv_args)(x)
    x = tf.keras.layers.Conv2D(32, 3, **conv_args)(x)
    x = tf.keras.layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return tf.keras.Model(inputs, outputs)

# Getting the low-resolution image
def get_lowres_image(img):
    image = tf.image.resize(img, (64, 60))
    return image

# Predict the image based off the model
def predict_image(model, img):
    image = model.predict(img[tf.newaxis, ...])
    return image[0]