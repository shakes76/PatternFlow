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
