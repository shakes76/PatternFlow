import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf


# vector quantizer class
class VQ(layers.Layer):
    def __init__(self, embed_n, embed_d, beta=0.25, **kwargs):
        super().__init__(**kwargs)
        self.embed_d = embed_d
        self.embed_n = embed_n
        self.beta = beta

        # fnitialise embeddings to be quantized
        w_init = tf.random_uniform_initializer()
        self.embeds = tf.Variable(
            initial_value=w_init(shape=(self.embed_d, self.embed_n), dtype="float32"),
            trainable=True,
            name="vqvae-embeddings",
        )

    def call(self, x):
        # flatten inputs while maintaining embed_d then quantize
        shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embed_d])
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.embed_n)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # get back original shape
        quantized = tf.reshape(quantized, shape)

        # loss
        c_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        cb_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * c_loss + cb_loss)

        quantized = x + tf.stop_gradient(quantized - x)
        return quantized

    def get_indices(self, flattened):
        # l2-normalised distance between input and codes
        similarity = tf.matmul(flattened, self.embeds)
        dists = (
            tf.reduce_sum(flattened**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeds**2, axis=0)
            - 2*similarity
        )

        # get best indices
        encode_indices = tf.argmin(dists, axis=1)
        return encode_indices

def get_encoder(dim=16):
    inputs = keras.Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(
        inputs
    )
    x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
    out = layers.Conv2D(dim, 1, padding="same")(x)
    return keras.Model(inputs, out, name="encoder")


def get_decoder(dim=16):
    inputs = keras.Input(shape=get_encoder(dim).output.shape[1:])
    x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(
        inputs
    )
    x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
    out = layers.Conv2DTranspose(1, 3, padding="same")(x)
    return keras.Model(inputs, out, name="decoder")


