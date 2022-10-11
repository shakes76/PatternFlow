"""
    Define all modules and components of the VQ-VAE and PixelCNN.

    Author: Adrian Rahul Kamal Rajkamal
    Student Number: 45811935
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np


def get_codebook_indices(encoded, inputs):
    """
        Return the codebook indices of the encoding 'closest' (defined by a normalised Euclidean
        norm) to the input (used for both PixelCNN and VQ-VAE).

        Args:
            encoded: Encoded vectors
            inputs: Input (flattened to vectors)

        Returns: index of closest encoding, with respect to normalised Euclidean norm
    """
    # Calculate 'closeness' between the (flattened) inputs and the encodings.
    norms = (
            tf.reduce_sum(inputs ** 2, axis=1, keepdims=True) +
            tf.reduce_sum(encoded ** 2, axis=0) -
            2 * tf.linalg.matmul(inputs, encoded)
    )
    return tf.argmin(norms, axis=1)


class VQ(layers.Layer):
    """
        Define Vector-Quantisation (VQ) Layer for VQ-VAE.
    """

    def __init__(self, num_encoded, encoded_dim, beta=0.25, layer_name="VectorQuantiser"):
        super().__init__(layer_name=layer_name)

        self._encoded_dim = encoded_dim
        self._num_encoded = num_encoded
        self._beta = beta

        # Provide a Uniform Distribution Prior for encoded vectors
        runif_initialiser = tf.random_uniform_initializer()
        encoded_shape = self._encoded_dim, self._num_encoded
        self._encoded = tf.Variable(initial_value=runif_initialiser(shape=encoded_shape,
                                                                    dtype="float32"))

    def get_encoded(self):
        """ Return encoded vectors """
        return self._encoded

