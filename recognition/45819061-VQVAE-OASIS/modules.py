from base64 import decode
from matplotlib.cbook import flatten
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Layer, ReLU, Add, Conv2D, Conv2DTranspose


class VectorQuantizer(Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, name="VQ"):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # Initialise flattened embeddings
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype='float32'),
            trainable=True,
            name=name
        )

    def call(self, x):
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, (-1, self.embedding_dim))

        # Quantization
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        quantized = tf.reshape(quantized, input_shape)

        commitment_loss = tf.norm(tf.stop_gradient(quantized) - x)**2
        codebook_loss = tf.norm(tf.stop_gradient(x) - quantized)**2
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        quantized = x + tf.stop_gradient(quantized - x)
        return quantized
        
    def get_code_indices(self, flattened_inputs):
        similarity = tf.matmul(flattened_inputs, self.embeddings)
        distances = (
            tf.reduce_sum(flattened_inputs**2, axis=1, keepdims=True)
            + tf.reduce_sum(self.embeddings**2, axis=0)
             - 2 * similarity
        )

        encoding_indices = tf.argmin(distances, axis=1)
        return encoding_indices


def resblock(x, filters=256):
    skip = Conv2D(filters, 1, strides=1, padding='same')(x)
    x = Conv2D(filters, 3, strides=1, padding='same')(x)
    x = ReLU()(x)
    x = Conv2D(filters, 1, strides=1, padding='same')(x)
    out = Add()([x, skip])
    return ReLU()(out)

class VQVAE(tf.keras.Model):
    def __init__(self, latent_dim=32, num_embeddings=64, input_shape=(256, 256, 1), residual_hiddens=64):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Build encoder
        encoder_in = Input(shape=input_shape)
        x1 = Conv2D(32, 4, strides=2, activation='leaky_relu', padding='same')(encoder_in)
        x2 = Conv2D(64, 4, strides=2, activation='leaky_relu', padding='same')(x1)
        x3 = resblock(x2, residual_hiddens)
        x4 = resblock(x3, residual_hiddens)
        encoder_out = Conv2D(latent_dim, 1, padding="same")(x4)
        self.encoder = tf.keras.Model(encoder_in, encoder_out, name='encoder')

        # Build decoder
        decoder_in = Input(shape=self.encoder.output.shape[1:])
        y1 = resblock(decoder_in, residual_hiddens)
        y2 = resblock(y1, residual_hiddens)
        y3 = Conv2DTranspose(64, 4, strides=2, activation='leaky_relu', padding='same')(y2)
        decoder_out = Conv2DTranspose(1, 4, strides=2, activation='leaky_relu', padding='same')(y3)
        self.decoder = tf.keras.Model(decoder_in, decoder_out, name='decoder')

        # Add VQ layer
        self.vq_layer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim, name='vq')

        #self.summary()
        

    def call(self, x, training=False):
        x = self.encoder(x)
        quantized = self.vq_layer(x)
        return self.decoder(quantized)
        

        