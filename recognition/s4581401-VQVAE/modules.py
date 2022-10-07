import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras


class VQVAE(tf.keras.Model):
    def __init__(self, latent_dim, img_shape):
        self._encoder = self.set_encoder(latent_dim, img_shape)

    def set_encoder(self, latent_dim, img_shape):

        encoder_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=img_shape),
            tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2), padding="same"),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2), padding="same"),
            tf.keras.layers.Conv2D(latent_dim, (1, 1), padding="same")
        ])
        return encoder_model

    def get_encoder(self):
        return self._encoder

    def set_decoder(self, img_shape):
        decoder_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=img_shape),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D((2, 2), padding="same"),
            tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D((2, 2), padding="same"),
            tf.keras.layers.Conv2D(1, (3, 3), padding="same")
        ])
        return decoder_model


