from typing import Tuple

import keras
import keras.layers
import keras.models
import tensorflow as tf
import numpy as np

# VQ VAE Model

def create_encoder_model(latent_dimensions: int,
        input_shape: Tuple[int, int, int]) -> keras.models.Sequential:
    '''Returns the Encoder model used for the VQ VAE.'''
    return keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(
                32, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2D(
                64, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2D(
                128, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2D(
                128, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2D(latent_dimensions, 1, padding="same")
    ], name="encoder")

def create_decoder_model(input_shape: Tuple[int, int, int]) \
        -> keras.models.Sequential:
    '''Returns the Decoder model used for the VQ VAE.'''
    return keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2DTranspose(
                128, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2DTranspose(
                128, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2DTranspose(
                64, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2DTranspose(
                32, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2DTranspose(
                3, 3, padding="same")
    ], name="decoder")

def create_vqvae_model(latent_dimensions: int, number_of_embeddings: int,
        input_shape: Tuple[int, int, int]) -> keras.models.Sequential:
    '''
    Combines an Encoder and Decoder with a VectorQuantizer layer to make a
    VQ VAE.
    '''
    encoder = create_encoder_model(latent_dimensions, input_shape)
    decoder = create_decoder_model(encoder.output_shape[1:])
    return keras.models.Sequential([
        keras.layers.Input(shape=input_shape),
        encoder,
        VectorQuantizer(number_of_embeddings, latent_dimensions,
                name="vector_quantizer"),
        decoder,
    ])

class VectorQuantizer(keras.layers.Layer):
    '''
    A layer that takes a batch of images and quantizes an embedding based on
    this.
    '''
    def __init__(self, number_of_embeddings: int, embedding_dimensions,
            beta: int = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.number_of_embeddings = number_of_embeddings
        self._embedding_dimensions = embedding_dimensions
        self._beta = beta

        self.embeddings = tf.Variable(
            initial_value=tf.random_uniform_initializer()(
                shape=(self._embedding_dimensions, self.number_of_embeddings),
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
        encodings = self.encode(flat)
        encodings = tf.one_hot(encodings, self.number_of_embeddings)

        # Quantize the given input based on the generated embeddings
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        quantized_original_dims = tf.reshape(quantized, tf.shape(x))

        # Calculate the loss for this layer based on VQ objective and
        # "commitment loss" used to stop the embeddings from growing given.
        # Both of these losses are given in the original VQ VAE paper.
        loss = tf.reduce_mean(
                (tf.stop_gradient(x) - quantized_original_dims) ** 2) \
                + self._beta * tf.reduce_mean((x - tf.stop_gradient(
                quantized_original_dims)) ** 2)
        self.add_loss(loss)

        quantized_original_dims = x + tf.stop_gradient(
                quantized_original_dims - x)
        return quantized_original_dims

    def encode(self, x):
        distances = tf.reduce_sum(x**2, 1, keepdims=True) \
                - 2 * tf.matmul(x, self.embeddings) \
                + tf.reduce_sum(self.embeddings**2, 0, keepdims=True)
        return tf.argmin(distances, axis=1)


# PixelCNN Prior Model

def create_pixel_cnn(latent_width, latent_height, num_embeddings)\
        -> keras.models.Model:
    '''
    Creates the PixelCNN model with 3 MaskedConv2D layers, 2 PixelCnnBlocks
    and a single Conv2D layer for output.
    '''
    input = keras.layers.Input(
            shape=(latent_height, latent_width), dtype=tf.int32)
    one_hot = tf.one_hot(input, num_embeddings)
    x = MaskedConv2D(mask_type="A", filters=128, kernel_size=7,
            activation="relu", padding="same")(one_hot)
    x = PixelCnnBlock(filters=128, kernel_size=3, is_first=False)(x)
    x = PixelCnnBlock(filters=128, kernel_size=3, is_first=False)(x)
    x = MaskedConv2D(mask_type="B", filters=128, kernel_size=1,
            strides=1, activation="relu", padding="valid")(x)
    x = MaskedConv2D(mask_type="B", filters=128, kernel_size=1,
            strides=1, activation="relu", padding="valid")(x)
    x = keras.layers.Conv2D(filters=num_embeddings, kernel_size=1,
            strides=1, padding="valid")(x)
    return keras.models.Model(input, x)

class MaskedConv2D(keras.layers.Layer):
    '''
    The MaskedConv2D layer is a standard Conv2D layer with a mask applied
    on top of it to treat the input as a sequence of data.
    '''
    def __init__(self, mask_type, **kwargs):
        super(MaskedConv2D, self).__init__()
        self.mask_type = mask_type
        self.conv = keras.layers.Conv2D(**kwargs)

    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel_shape = tf.shape(self.conv.kernel)

        center_h = kernel_shape[0] // 2
        center_w = kernel_shape[1] // 2
        
        mask = tf.Variable(tf.zeros(shape=kernel_shape))
        mask_shape = mask[: center_h, ...].shape
        mask[: center_h, ...].assign(tf.ones(shape=mask_shape))
        mask_shape = mask[center_h, : center_w, ...].shape
        mask[center_h, : center_w, ...].assign(tf.ones(shape=mask_shape))
        if self.mask_type == "B":
            mask_shape = mask[center_h, center_w, ...].shape
            mask[center_h, center_w, ...].assign(tf.ones(shape=mask_shape))

        self.mask = tf.constant(mask, dtype=tf.float32)

    def call(self, input):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(input)

class PixelCnnBlock(keras.models.Model):
    '''
    A PixelCnnBlock contains a MaskedConv2D layer with a convolution
    before and after it is applied. This Block is also residual,
    meaning that the input is added to the output.
    '''
    def __init__(self, filters, kernel_size, is_first: bool = False):
        super(PixelCnnBlock, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.is_first = is_first

        self.pre_conv = keras.layers.Conv2D(
                filters=self.filters, kernel_size=1, activation="relu")
        masked_conv_type = ("A" if is_first else "B")
        self.masked_conv = MaskedConv2D(
                mask_type=masked_conv_type,
                filters=self.filters // 2,
                kernel_size=kernel_size, activation="relu", padding="same")
        self.post_conv = keras.layers.Conv2D(
                filters=self.filters, kernel_size=1, activation="relu")

    def call(self, inputs):
        pre_conv_applied = self.pre_conv(inputs)
        mask_applied = self.masked_conv(pre_conv_applied)
        post_conv_applied = self.post_conv(mask_applied)
        # This block is residual so add the input value to the result
        return keras.layers.add([inputs, post_conv_applied])
