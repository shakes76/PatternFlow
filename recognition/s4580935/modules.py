import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

class VectorQuantizer(layers.Layer):
    """
    Create the custom VectorQuantizer class layer, which is inbetween the encoder and decoder
    """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, **kwargs):
        """
        Initialise the layer

        Parameters:
        num_embeddings: number of embeddings in the codebook
        embedding_dim: The dimensionality of the embedding vector
        beta: Used to calculate loss, initiates between [0.25, 2] per paper
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.beta = beta

        #Initialise embeddings to be quantized
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embedding_dim, self.num_embeddings), dtype="float32"
            ),
            trainable=True,
            name="embeddings_vqvae",
        )
    
    def call(self, given):
        # Calculate the shape of the given input
        input_shape = tf.shape(given)
        # flatten the given input keeping dimensionality intact.
        flattened = tf.reshape(given, [-1, self.embedding_dim])

        # Quantization.
        encoding_indices = self.get_code_indices(flattened)
        encodings = tf.one_hot(encoding_indices, self.num_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)

        # Reshape the quantized values back to the original input shape
        quantized = tf.reshape(quantized, input_shape)

        # Calculate vector quantization loss and add that to the layer. 
        commitment_loss = tf.reduce_mean((tf.stop_gradient(quantized) - given) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(given)) ** 2)
        self.add_loss(self.beta * commitment_loss + codebook_loss)

        # Straight-through estimator.
        quantized = given + tf.stop_gradient(quantized - given)
        return quantized
