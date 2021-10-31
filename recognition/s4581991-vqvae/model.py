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
        keras.layers.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(
                32, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2D(
                64, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2D(latent_dimensions, 1, padding="same")
    ], name="encoder")

def create_decoder_model(input_shape: Tuple[int, int, int]) \
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
        keras.layers.Input(shape=(28, 28, 1)),
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
        # Both of these losses are given in the original VQ VAE paper
        # <add reference>.
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

def create_pixel_cnn(height, width, num_embeddings) \
        -> keras.Model:
    input = keras.layers.Input(shape=(7, 7), dtype=tf.int32)
    one_hot = tf.one_hot(input, num_embeddings)
    v, h = GatedPixelCnnBlock(filters=128, kernel_size=1, is_first=True)([one_hot, one_hot])
    v, h = GatedPixelCnnBlock(filters=128, kernel_size=1, is_first=False)([v, h])
    v, h = GatedPixelCnnBlock(filters=128, kernel_size=1, is_first=False)([v, h])
    v, h = GatedPixelCnnBlock(filters=128, kernel_size=1, is_first=False)([v, h])
    v, h = GatedPixelCnnBlock(filters=128, kernel_size=1, is_first=False)([v, h])
    x = keras.layers.Activation(activation="relu")(h)
    x = keras.layers.Conv2D(filters=128, kernel_size=1, strides=1)(x)
    x = keras.layers.Activation(activation="relu")(x)
    x = keras.layers.Conv2D(filters=num_embeddings, kernel_size=1, strides=1, padding="valid")(x)
    return keras.models.Model(input, x)

class MaskedConv2D(keras.layers.Layer):
    def __init__(self, mask_type, **kwargs):
        super(MaskedConv2D, self).__init__()
        self.mask_type = mask_type
        self.conv = keras.layers.Conv2D(**kwargs)

    def build(self, input_shape):
        self.conv.build(input_shape)
        kernel_shape = tf.shape(self.conv.kernel)
        mask = np.ones(shape=kernel_shape)
        # print(mask.shape)
        center_h = kernel_shape[0] // 2
        center_w = kernel_shape[1] // 2
        if self.mask_type == "V":
            mask[center_h + 1:, :, :, :] = 0.0
        else:
            mask[:center_h, :, :] = 0.0
            mask[center_h + 1:, :, :] = 0.0
            if self.mask_type == "A":
                mask[center_h, center_w:, :, :] = 0.0
            elif self.mask_type == "B":
                mask[center_h, center_w + 1:, :, :] = 0.0
        # print("A")
        # print(mask.shape)
        self.mask = tf.constant(mask, dtype=tf.float32)

    def call(self, input):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        # print("B")
        # print(input.shape)
        # print(self.conv.kernel.shape)
        # print(self.conv(input).shape)
        return self.conv(input)

class GatedPixelCnnBlock(keras.models.Model):
    def __init__(self, filters, kernel_size, is_first: bool = False):
        super(GatedPixelCnnBlock, self).__init__()

        self.filters = filters
        self.kernel_size = kernel_size
        self.is_first = is_first

        self.vertical_conv = MaskedConv2D(mask_type="V",
                filters=2 * filters, kernel_size=kernel_size)
        self.horizontal_conv = MaskedConv2D(
                mask_type=("A" if is_first else "B"), filters=2 * filters,
                kernel_size=kernel_size)

        self.v_to_h_conv = keras.layers.Conv2D(
                filters=2 * self.filters, kernel_size=1, strides=(1,1),
                padding="same")
        self.horizontal_after_gate_conv = keras.layers.Conv2D(
            filters=self.filters, kernel_size=1, strides=(1,1), padding="same")

        self.vertical_top_padding = keras.layers.ZeroPadding2D(
                padding=((1, 0), (0, 0)))
        self.vertical_bottom_crop = keras.layers.Cropping2D(
                cropping=((0, 1), (0, 0)))

    def _split_feature_maps(self, x):
        return tf.split(x, 2, axis=-1)

    def _apply_gate_activation(self, x):
        split = self._split_feature_maps(x)
        return tf.math.multiply(
            tf.math.tanh(split[0]),
            tf.math.sigmoid(split[1])
        )

    def call(self, inputs):
        vertical_stack_in = inputs[0]
        horizontal_stack_in = inputs[1]

        vertical_before_gate = self.vertical_conv(vertical_stack_in)
        vertical_stack_out = self._apply_gate_activation(vertical_before_gate)
        # print(vertical_before_gate.shape)
        vertical_for_horizontal = self.vertical_top_padding(
                vertical_before_gate)
        vertical_for_horizontal = self.vertical_bottom_crop(
                vertical_for_horizontal)
        vertical_for_horizontal = self.v_to_h_conv(vertical_before_gate)

        horizontal_before_gate = self.horizontal_conv(horizontal_stack_in)
        horizontal_before_gate = vertical_for_horizontal \
                    + horizontal_before_gate

        horizontal_stack_out = self._apply_gate_activation(
                horizontal_before_gate)
        horizontal_stack_out = self.horizontal_after_gate_conv(
                horizontal_stack_out)

        # print(horizontal_stack_out.shape)
        # print(vertical_stack_out.shape)


        # Add residual if this isn't the first block in the network
        if self.is_first:
            return vertical_stack_out, horizontal_stack_out

        return vertical_stack_out, keras.layers.add(
                [horizontal_stack_in, horizontal_stack_out])
