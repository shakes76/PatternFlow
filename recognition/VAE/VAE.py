import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, InputLayer, Flatten, Dense, Reshape, BatchNormalization
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from skimage import data, img_as_float
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
import os
from tensorflow.keras.models import load_model
import math


class VAE(tf.keras.Model):
    def __init__(self, latent_dimsion):
        super(VAE, self).__init__()
        self.latent_dim = latent_dimsion
        self.encoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(256, 256, 1)),
                Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu'),
                BatchNormalization(),
                Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
                BatchNormalization(),
                Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
                BatchNormalization(),
                Conv2D(filters=128, kernel_size=3, strides=(2, 2), activation='relu'),
                BatchNormalization(),
                Flatten(),
                # No activation
                Dense(latent_dimsion * 2),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                InputLayer(input_shape=(latent_dimsion,)),
                Dense(units=32 * 32 * 32, activation=tf.nn.relu),
                BatchNormalization(),
                Reshape(target_shape=(32, 32, 32)),
                Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                Conv2DTranspose(filters=16, kernel_size=3, strides=2, padding='same', activation='relu'),
                BatchNormalization(),
                # No activation
                Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same'),
            ]
        )
