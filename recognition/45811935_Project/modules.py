"""
    Defines all modules and components of the VQ-VAE and PixelCNN.

    Author: Adrian Rahul Kamal Rajkamal
    Student Number: 45811935
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers, Model

import numpy as np

""" VQ-VAE """


class VQ(layers.Layer):
    """
        Define custom Vector-Quantisation (VQ) Layer for VQ-VAE.
    """

    def __init__(self, num_encoded, latent_dim, beta=0.25, layer_name="vq"):
        super().__init__(layer_name=layer_name)

        self._latent_dim = latent_dim
        self._num_encoded = num_encoded
        self._beta = beta

        # Provide a Uniform Distribution Prior for encoded vectors
        runif_initialiser = tf.random_uniform_initializer()
        encoded_shape = self._latent_dim, self._num_encoded
        self._encoded = tf.Variable(initial_value=runif_initialiser(shape=encoded_shape,
                                                                    dtype="float32"))

    def get_encoded(self):
        """ Returns encoded vectors """
        return self._encoded

    def get_codebook_indices(self, inputs):
        """
            Returns the codebook indices of the encodings 'closest' (defined by a normalised
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
    """
        Defines Encoder for VQ-VAE.
    """
    def __init__(self, img_size=28, latent_dim=16, name="encoder"):
        super(Encoder, self).__init__(name=name)
        self._latent_dim = latent_dim
        self._img_size = img_size

        self.input1 = layers.InputLayer(input_shape=self._img_size)
        self.conv1 = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")
        self.conv2 = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")
        self.conv3 = layers.Conv2D(self._latent_dim, 1, padding="same")

    def call(self, inputs):
        """
            Forward computation of this layer.

            Args:
                inputs: inputs of this layer

            Returns: outputs of this layer
        """
        inputs = self.input1(inputs)
        hidden = self.conv1(inputs)
        hidden = self.conv2(hidden)
        return self.conv3(hidden)


class Decoder(Model):
    """
        Defines Decoder for VQ-VAE.
    """
    def __init__(self, input_dim=16, name="decoder"):
        super(Decoder, self).__init__(name=name)
        self.input1 = layers.InputLayer(input_shape=input_dim)
        self.conv_t1 = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")
        self.conv_t2 = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")
        self.conv_t3 = layers.Conv2DTranspose(1, 3, padding="same")

    def call(self, inputs):
        """
            Forward computation of this layer.

            Args:
                inputs: inputs of this layer

            Returns: outputs of this layer
        """
        inputs = self.input1(inputs)
        hidden = self.conv_t1(inputs)
        hidden = self.conv_t2(hidden)
        return self.conv_t3(hidden)


class VQVAE(Model):
    """
        Defines main VQ-VAE architecture.
    """
    def __init__(self, tr_var, img_size=28, num_encoded=64, latent_dim=16,
                 beta=0.25, name="vq_vae"):
        super(VQVAE, self).__init__(name=name)
        self._tr_var = tr_var
        self._img_size = img_size
        self._num_encoded = num_encoded
        self._latent_dim = latent_dim
        self._beta = beta

        self._encoder = Encoder(self._img_size, self._latent_dim)
        self._vq = VQ(self._num_encoded, self._latent_dim, self._beta)
        self._decoder = Decoder(self._encoder.output.shape[1:])

        self._total_loss = tf.keras.metrics.Mean(name="total_loss")
        self._vq_loss = tf.keras.metrics.Mean(name="vq_loss")
        self._reconstruction_loss = tf.keras.metrics.Mean(name="reconstruction_loss")

    def call(self, inputs):
        """
            Forward computation of this layer.

            Args:
                inputs: inputs of this layer

            Returns: outputs of this layer
        """
        encoded = self._encoder(inputs)
        quantised = self._vq(encoded)
        decoded = self._decoder(quantised)
        return decoded

    def train_step(self, data):
        """
            Performs one iteration of training and return loss values.

            Args:
                data: input data

            Returns: loss values

        """
        with tf.GradientTape() as tape:
            # Get reconstructions
            recon = self.vqvae(data)

            # Obtain loss values
            reconstruction_loss_val = (
                    tf.reduce_mean((data - recon) ** 2) / self._tr_var
            )
            total_loss_val = reconstruction_loss_val + sum(self.get_vq().losses)

        # Perform Backpropagation
        grads = tape.gradient(total_loss_val, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Update loss values
        self.total_loss.update_state(total_loss_val)
        self.reconstruction_loss.update_state(reconstruction_loss_val)
        self.vq_loss.update_state(sum(self.get_vq().losses))

        # Log results.
        return {
            "loss": self.total_loss.result(),
            "vq_loss": self.vq_loss.result(),
            "reconstruction_loss": self.reconstruction_loss.result(),
        }

    @property
    def metrics(self):
        """ Returns metrics """
        return [
            self.total_loss,
            self.vq_loss,
            self.reconstruction_loss,
        ]

    def get_encoder(self):
        """ Returns encoder """
        return self._encoder

    def get_vq(self):
        """ Returns VQ layer """
        return self._vq

    def get_decoder(self):
        """ Returns decoder """
        return self._decoder
