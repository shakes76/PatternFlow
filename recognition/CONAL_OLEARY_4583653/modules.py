from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Reference to https://keras.io/examples/generative/vq_vae/


class VectorQuantizer(layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.2, **kwargs):
        """
          num_embeddings: The number of embeddings
          embedding_dim: The dimension of the embedding vector
          beta: Loss factor 
        """
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        initialiser = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=initialiser(
                shape=(self.embedding_dim,
                       self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="VQVAE_EMBEDDINGS",
        )

    def call(self, input):

        # Flattening
        shape = tf.shape(input)
        flattened_input = tf.reshape(input, [-1, self.embedding_dim])

        # Quantizing
        encoding_indices = self.get_code_indices(flattened_input)
        encodings = tf.one_hot(
            encoding_indices, self.num_embeddings, dtype="float32")
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized = tf.reshape(quantized, shape)

        # Losses
        commitment_loss = tf.reduce_mean(
            (tf.stop_gradient(quantized) - input) ** 2)
        codebook_loss = tf.reduce_mean(
            (quantized - tf.stop_gradient(input)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = input + tf.stop_gradient(quantized - input)
        return quantized

    def get_code_indices(self, flattened_input):
        similarity = tf.matmul(flattened_input, self.embeddings)
        distances = (tf.reduce_sum(flattened_input**2, axis=1, keepdims=True) +
                     tf.reduce_sum(self.embeddings**2, axis=0) - 2*similarity)

        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


class Encoder(keras.models.Model):
    def __init__(self, latent_dim=256, **kwargs):
        """
          latent_dim: Dimension of the latent output space
        """
        super().__init__(**kwargs)
        self.intermediate_layers = [
            layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
            layers.Conv2D(128, 3, activation="relu",
                          strides=2, padding="same"),
        ]
        self.final_layer = layers.Conv2D(latent_dim, 1, padding="same")

    def call(self, input):
        payload = input
        for layer in self.intermediate_layers:
            payload = layer(payload)
        final_result = self.final_layer(payload)
        return final_result


class Decoder(keras.models.Model):
    def __init__(self, latent_dim=256, **kwargs):
        """
          latent_dim: Dimension of the latent input space
        """
        super().__init__(**kwargs)
        self.intermediate_layers = [
            layers.Conv2DTranspose(latent_dim, 3, activation="relu",
                                   strides=2, padding="same"),
            layers.Conv2DTranspose(latent_dim // 2, 3, activation="relu",
                                   strides=2, padding="same"),
            layers.Conv2DTranspose(latent_dim // 4, 3, activation="relu",
                                   strides=2, padding="same"),
        ]
        self.final_layer = layers.Conv2DTranspose(1, 3, padding="same")

    def call(self, input):
        payload = input
        for layer in self.intermediate_layers:
            payload = layer(payload)
        final_result = self.final_layer(payload)
        return final_result
