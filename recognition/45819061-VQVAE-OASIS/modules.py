from matplotlib.cbook import flatten
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, Flatten, Reshape


class VectorQuantizer(tf.layers.Layer):
    def __init__(self, num_embeddings, embedding_dim, beta=0.25):
        super.__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta

        # Initialise embeddings
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(shape=(self.embedding_dim, self.num_embeddings), dtype='float64'),
            trainable=True,
            name="VQ"
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

class VQVAE(tf.keras.Model):
    def __init__(self, latent_dim=32, num_embeddings=64, input_shape=(256, 256, 3)):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Build encoder
        encoder_in = Input(shape=input_shape)
        x = Conv2D(32, 3, strides=2, activation='leakyrelu', padding='same')(encoder_in)
        x = Conv2D(64, 3, strides=2, activation='leakyrelu', padding='same')(x)
        x = Conv2D(64, 3, strides=2, activation='leakyrelu', padding='same')(x)
        encoder_out = Conv2D(latent_dim, 1, padding='same')(x)
        self.encoder = tf.keras.Model(encoder_in, encoder_out, name='encoder')

        # Build decoder
        decoder_in = Input(shape=self.encoder.output.shape[1:])
        x = Conv2DTranspose(64, 3, strides=2, activation='leakyrelu', padding='same')(encoder_in)
        x = Conv2DTranspose(64, 3, strides=2, activation='leakyrelu', padding='same')(x)
        x = Conv2DTranspose(32, 3, strides=2, activation='leakyrelu', padding='same')(x)
        decoder_out = Conv2DTranspose(1, 3, padding='same')(x)
        self.decoder = tf.keras.Model(decoder_in, decoder_out, name='decoder')

        # Add VQ layer
        self.vq_layer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=latent_dim, name='vq')
        

    def call(self, x, training=False):
        x = self.encoder(x)
        quantized = self.vq_layer(x)
        return self.decoder(quantized)
        

        