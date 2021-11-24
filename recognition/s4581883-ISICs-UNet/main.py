import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pathlib
import keras
from tensorflow import keras
from keras import layers, preprocessing
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, LeakyReLU, Dropout, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()


def dice_coefficient(true, pred, axis = (1,2,3)):
    intersect = (2 * tf.reduce_sum(true * pred, axis))
    union = tf.reduce_sum(true, axis) + tf.reduce_sum(pred, axis)
    dice_coeff = tf.reduce_mean(intersect / union)
    return dice_coeff

def dice_coefficient_loss(true, pred):
    return 1 - dice_coefficient(true, pred)

