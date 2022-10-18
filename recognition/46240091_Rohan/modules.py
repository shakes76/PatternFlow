
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
from keras import layers


def encoder(latent_dim=16):
    model = keras.Sequential();
    model.add(keras.Input(shape=(128, 128, 1)))
    model.add(layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"))
    model.add(layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"))
    model.add(layers.Conv2D(latent_dim, 1, padding="same"))
    return model


def decoder(latent_dim=16):
    model = keras.Sequential();
    model.add(keras.Input(shape=encoder().output.shape[1:]))
    model.add(layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"))
    model.add(layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"))
    model.add(layers.Conv2DTranspose(1, 3, padding="same"))
    return model

#Vector Quantizer custom layer according to https://keras.io/examples/generative/vq_vae/#vectorquantizer-layer
class VectorQuantizer(keras.layers.Layer):

    def __init__(self, num_embeddings, embedding_dim, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = 0.25
        
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae",
        )

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, input_shape)
        commitment_loss = self.beta * tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(commitment_loss + codebook_loss)
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )
        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices