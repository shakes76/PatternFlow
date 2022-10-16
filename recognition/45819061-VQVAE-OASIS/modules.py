from matplotlib.cbook import flatten
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Conv2DTranspose, Flatten, Reshape


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

