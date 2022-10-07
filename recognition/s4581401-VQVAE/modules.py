import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras

###
# References:
# - https://keras.io/examples/generative/vq_vae/
# - https://www.kaggle.com/code/ameroyer/keras-vq-vae-for-image-generation/notebook
##


class VectorQuantizerLayer(tf.keras.layers.Layer):
    def __init__(self, embedding_num, embedding_dim, beta, **kwargs):
        super().__init__(**kwargs)
        self._embedding_num = embedding_num
        self._embedding_dim = embedding_dim
        self._beta = beta

        #Initialise

class VQVAEModel(tf.keras.Model):
    def __init__(self, latent_dim, img_shape, embedding_num, embedding_dim, beta, **kwargs):

        super(VQVAEModel, self).__init__(**kwargs)

        self._latent_dim = latent_dim
        self._img_shape = img_shape
        self._encoder = self.set_encoder(latent_dim, img_shape)
        self._decoder = self.set_decoder(img_shape)
        self._embedding_num = embedding_num
        self._embedding_dim = embedding_dim
        self._beta = beta

    @staticmethod
    def set_encoder(latent_dim, img_shape):
        encoder_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=img_shape),
            tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2), padding="same"),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D((2, 2), padding="same"),
            tf.keras.layers.Conv2D(latent_dim, (1, 1), padding="same")
        ], name="encoder")
        return encoder_model

    def get_encoder(self):
        return self._encoder

    @staticmethod
    def set_decoder(img_shape):
        decoder_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=img_shape),
            tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D((2, 2), padding="same"),
            tf.keras.layers.Conv2D(16, (3, 3), padding="same", activation="relu"),
            tf.keras.layers.UpSampling2D((2, 2), padding="same"),
            tf.keras.layers.Conv2D(1, (3, 3), padding="same")
        ], name="decoder")
        return decoder_model

    def get_decoder(self):
        return self._decoder


