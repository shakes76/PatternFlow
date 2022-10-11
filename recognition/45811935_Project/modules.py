"""
    Define all modules and components of the VQ-VAE and PixelCNN.

    Author: Adrian Rahul Kamal Rajkamal
    Student Number: 45811935
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

import numpy as np


class VQ(layers.Layer):
    """
        Define custom Vector-Quantisation (VQ) Layer for VQ-VAE.
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

    def get_codebook_indices(self, inputs):
        """
            Return the codebook indices of the encodings 'closest' (defined by a normalised
            Euclidean norm) to the input.

            Args:
                inputs: Input (flattened to vectors)

            Returns: indices of the closest encodings, with respect to normalised Euclidean norm
        """
        # Calculate 'closeness' between the (flattened) inputs and the encodings.
        norms = (
                tf.reduce_sum(inputs ** 2, axis=1, keepdims=True) +
                tf.reduce_sum(self._encoded ** 2, axis=0) -
                2 * tf.linalg.matmul(inputs, self._encoded)
        )
        return tf.argmin(norms, axis=1)

    def call(self, inputs):
        """
            Forward computation of this layer.

            Args:
                inputs: inputs of this layer

            Returns: outputs of this layer

        """
        # Flatten input and store original dimensions for reshaping later
        original_shape = tf.shape(inputs)
        flattened_input = tf.reshape(inputs, [-1, self._encoded_dim])

        # Perform quantization (i.e. compression)
        encoded_indices = self.get_codebook_indices(flattened_input)
        onehot_indices = tf.one_hot(encoded_indices, self._num_encoded)
        quantized = tf.reshape(
            tf.linalg.matmul(onehot_indices, self._encoded, transpose_b=True),
            original_shape
        )

        # Calculate vector quantization loss from [1] in README.md. The stop_gradient function
        # treats its input as a constant during forward computation, i.e. stopping the computation
        # of its gradient, as it would effectively be 0, hence detaching it from the computational
        # graph.
        quantized_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs)) ** 2)
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs) ** 2)
        self.add_loss(self._beta * commitment_loss + quantized_loss)  # Total loss in training loop

        # Return the (straight-through) estimator (i.e. an estimator that is treated like an
        # identity function with respect to gradients during backprop, and is detached from the
        # computational graph as a result)
        return inputs + tf.stop_gradient(quantized - inputs)


class Encoder(Model):
    def __init__(self):
        pass

    def call(self, inputs):
        pass


class Decoder(Model):
    def __init__(self):
        pass

    def call(self, inputs):
        pass


class VQVAE(Model):
    def __init__(self):
        pass

    def call(self, inputs):
        pass


