from typing import Tuple

import keras.layers
import keras.models
import tensorflow as tf

def create_encoder_model(latent_dimensions: int,
        input_shape: Tuple(int, int, int)) -> keras.models.Sequential:
    '''Returns the Encoder model used for the VQ VAE.'''
    return keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(
                32, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2D(
                64, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2D(latent_dimensions, 1, padding="same")
    ])

def create_decoder_model(input_shape: Tuple(int, int, int)) \
        -> keras.models.Sequential:
    '''Returns the Decoder model used for the VQ VAE.'''
    return keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2DTranspose(
                64, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2DTranspose(
                32, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2DTranspose(
                1, 3, padding="same")
    ])

def create_vqvae_model(latent_dimensions: int, number_of_embeddings: int,
        input_shape: Tuple(int, int, int)) -> keras.models.Sequential:
    '''
    Combines an Encoder and Decoder with a VectorQuantizer layer to make a
    VQ VAE.
    '''
    encoder = create_encoder_model(latent_dimensions, input_shape)
    decoder = create_decoder_model(encoder.output_shape)
    return keras.models.Sequential([
        encoder,
        VectorQuantizer(number_of_embeddings, latent_dimensions),
        decoder
    ])

class VectorQuantizer(keras.layers.Layer):
    '''
    A layer that takes a batch of images and quantizes an embedding based on
    this.
    '''
    def __init__(self, number_of_embeddings: int, embedding_dimensions,
            beta: int = 0.25):
        self.super.__init__()
        self._number_of_embeddings = number_of_embeddings
        self._embedding_dimensions = embedding_dimensions
        self._beta = beta

        self._embeddings = tf.Variable(
            initial_value=tf.random_uniform_initializer()(
                shape=(self._embedding_dimensions, self._number_of_embeddings),
                dtype="float32"
            ),
            trainable=True,
            name="embeddings"
        )

    def call(self, x):
        '''
        Calls this layer, calculating the output by quantizing the input vector
        which will be given by the encoder in the VQ VAE. Also calculates and
        stores the loss based on the encoder and decoder in the VQ VAE.
        '''
        flat = tf.reshape(x, [-1, self._embedding_dimensions])
        # Distances for each data point to the centres given by the embeddings
        distances = tf.reduce_sum(flat**2, 1, keepdims=True) \
                - 2 * tf.matmul(flat, self._embeddings) \
                + tf.reduce_sum(self._embeddings**2, 0, keepdims=True)
        encodings = tf.one_hot(tf.argmax(-distances, 1),
                self._number_of_embeddings, dtype=distances.dtype)

        # Quantize the given input based on the generated embeddings
        quantized = tf.matmul(encodings, self._embeddings, transpose_b=True)
        quantized_original_dims = tf.reshape(quantized, tf.shape(x))

        # Calculate the loss for this layer based on VQ objective and
        # "commitment loss" used to stop the embeddings from growing given.
        # Both of these losses are given in the original VQ VAE paper
        # <add reference>.
        loss = tf.reduce_mean((tf.stop_gradient(x) - self._embeddings) ** 2) \
                + self._beta * tf.reduce_mean((x - tf.stop_gradient(
                self._embeddings)) ** 2)
        self.add_loss(loss)

        quantized_original_dims = x + tf.stop_gradient(
                quantized_original_dims - x)
        return quantized_original_dims