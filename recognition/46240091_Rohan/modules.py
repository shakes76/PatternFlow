
from glob import glob
import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
import pathlib
import PIL
import PIL.Image
import tensorflow as tf
from tensorflow import keras
import os
from typing import Tuple
import keras.models
import keras.preprocessing.image
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import tensorflow_probability as tfp


def encoder(latent_dim=16):
    input_layer = keras.Input(shape=(128, 128, 1))
    e = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(input_layer)
    e = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(e)
    output = layers.Conv2D(latent_dim, 1, padding="same")(e)
    return keras.Model(input_layer, output, name="encoder")