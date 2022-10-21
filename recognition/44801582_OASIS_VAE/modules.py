import tensorflow as tf
from tensorflow import keras
import numpy as np


class VectorQuantizer(keras.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings

        self.beta = beta

        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype="float32"),
            trainable=True,
            name="embeddings_vqvae")

    def call(self, x):
        encoding_indices = self.get_code_indices(tf.reshape(x, [-1, self.embedding_dim]))
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.reshape(tf.matmul(encodings, self.embeddings, transpose_b=True), tf.shape(x))

        self.add_loss(self.beta * tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
                      + tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2))

        quantized = x + tf.stop_gradient(quantized - x)

        return quantized

    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        encoding_indices = tf.argmin((tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True)
                     + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * similarity), axis=1)

        return encoding_indices


def def_encoder(latent_dim):
    encoder_inputs = keras.Input(shape=(256, 256, 1))
    x = keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        encoder_inputs
    )
    x = keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    encoder_outputs = keras.layers.Conv2D(latent_dim, 1, padding="same")(x)
    return keras.Model(encoder_inputs, encoder_outputs, name="encoder")


def def_decoder(latent_dim):
    latent_inputs = keras.Input(shape=def_encoder(latent_dim).output.shape[1:])
    x = keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        latent_inputs
    )
    x = keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    decoder_outputs = keras.layers.Conv2DTranspose(1, 3, padding="same")(x)
    return keras.Model(latent_inputs, decoder_outputs, name="decoder")


def VQVAE(latent_dim=16, num_embeddings=64):
    vq_layer = VectorQuantizer(num_embeddings, latent_dim, name="vector_quantizer")
    encoder = def_encoder(latent_dim)
    decoder = def_decoder(latent_dim)
    inputs = keras.Input(shape=(256, 256, 1))
    encoded = encoder(inputs)
    quantized = vq_layer(encoded)
    reconstructions = decoder(quantized)
    return keras.Model(inputs, reconstructions, name="vq_vae")


class MaskedConvLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(MaskedConvLayer, self).__init__()
        self.convolution = keras.layers.Conv2D(**kwargs)

    def build(self, input_shape):
        self.convolution.build(input_shape)
        kernel_shape = self.convolution.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0

    def call(self, inputs):
        self.convolution.kernel.assign(self.convolution.kernel * self.mask)
        return self.convolution(inputs)


class ResidBlock(keras.layers.Layer):
    def __init__(self, filters, **kwargs):
        super(ResidBlock, self).__init__(**kwargs)
        self.c1 = keras.layers.Conv2D(filters, 1, activation="relu")
        self.mc = MaskedConvLayer(filters=filters // 2, kernel_size=3, activation="relu", padding="same")
        self.c2 = keras.layers.Conv2D(filters, 1, activation="relu")

    def call(self, inputs):
        x = self.c1(inputs)
        x = self.mc(x)
        x = self.c2(x)
        return keras.layers.add([inputs, x])


def PixelCNN(latent_dim, num_embeddings, num_residual_blocks, num_pixelcnn_layers):
    inputs = keras.Input(def_encoder(latent_dim).layers[-1].output_shape, dtype=tf.int32)
    encoding = tf.one_hot(inputs, num_embeddings)
    x = MaskedConvLayer(filters=128, kernel_size=7, activation="relu", padding="same")(encoding)

    for _ in range(num_residual_blocks):
        x = ResidBlock(128)(x)
    for _ in range(num_pixelcnn_layers):
        x = MaskedConvLayer(filters=128, kernel_size=1, activation="relu", padding="valid")(x)

    output = keras.layers.Conv2D(num_embeddings, 1, 1, padding="valid")(x)
    pixel_cnn = keras.Model(inputs, output)
    pixel_cnn.summary()

    return pixel_cnn

