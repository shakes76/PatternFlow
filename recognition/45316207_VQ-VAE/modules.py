"""
modules.py

Alex Nicholson (45316207)
11/10/2022

Contains the source code of the components of your model. Each component must be implementated as a class or a function

"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp


class VectorQuantizer(layers.Layer):
    """
    A Custom Vector Quantising Layer

        Attributes:
            name (str): first name of the person
            surname (str): family name of the person
            age (int): age of the person

        Methods:
            call(x): Calls the VectorQuantizer to quantise the input vector x???
            get_code_indices(flattened_inputs): Gets the indices of the codebook vectors???
    """

    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        # The `beta` parameter is best kept between [0.25, 2] as per the paper.
        self.beta = beta

        # Initialize the embeddings which we will quantize.
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )


    def call(self, x):
        """
        Calls the VectorQuantizer to quantise the input vector x???

            Parameters:
                self
                x (Tensorflow Tensor): An input vector to be quantised

            Returns:
                quantized (Tensorflow Tensor): The quantised version of the input vector???
        """

        # Calculate the input shape of the inputs and
        # then flatten the inputs keeping `embedding_dim` intact.
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. You can learn more
        # about adding losses to different layers here:
        # https://keras.io/guides/making_new_layers_and_models_via_subclassing/. Check
        # the original paper to get a handle on the formulation of the loss function.
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = x + tf.stop_gradient(quantized - x)
        return quantized


    def get_code_indices(self, flattened_inputs):
        """
        Gets the indices of the codebook vectors???

            Parameters:
                self
                flattened_inputs (Tensorflow Tensor): purpose???

            Returns:
                encoding_indices (Tensorflow Tensor): purpose???
        """

        # Calculate L2-normalized distance between the inputs and the codes.
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings ** 2, axis=0)
            - 2 * similarity
        )

        # Derive the indices for minimum distances.
        encoding_indices = tf.argmin(distances, axis=1)

        return encoding_indices


def get_encoder(latent_dim=16):
    """
    Encoder Module

        Parameters:
            (optional) latent_dim (int): The number of latent dimensions the images are compressed down to (default=16)

        Returns:
            encoder (Keras Model): The encoder module for the VQ-VAE
    """

    encoder_inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def get_decoder(latent_dim=16):
    """
    Decoder Module

        Parameters:
            (optional) latent_dim (int): The number of latent dimensions the images are compressed down to (default=16)

        Returns:
            decoder (Keras Model): The decoder module for the VQ-VAE
    """

    latent_inputs = keras.Input(shape=get_encoder(latent_dim).output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = layers.Conv2DTranspose(1, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


def get_vqvae(latent_dim=16, num_embeddings=64):
    """
    Builds the complete VQ-VAE Model out of its component modules

        Parameters:
            (optional) latent_dim (int): The number of latent dimensions the images are compressed down to (default=16)
            (optional) num_embeddings (int): The number of codebook vectors in the embedding space (default=64)

        Returns:
            vq_vae_model (Keras Model): A complete VQ-VAE model
    """

    print("Building model...")

    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = get_encoder(latent_dim)
    decoder = get_decoder(latent_dim)
    inputs = keras.Input(shape=(28, 28, 1))
    encoder_outputs = encoder(inputs)
    quantized_latents = vq_layer(encoder_outputs)
    reconstructions = decoder(quantized_latents)
    return keras.Model(inputs, reconstructions, name="vq_vae")


if __name__ == "__main__":
    model = get_vqvae()
    model.summary()