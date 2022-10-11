import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

class VQ(layers.Layer):
    
    def __init__(self, no_embeddings, embed_dim, beta=0.25):
        self.embed_dim = embed_dim
        self.no_embeddings = no_embeddings
        
        self.beta = beta
        
        w_init = tf.random_uniform_initializer()
        self.embeddings = tf.Variable(
            initial_value=w_init(
                shape=(self.embed_dim, self.no_embeddings), dtype='float32'),
            trainable=True, name='embedding_vq',)
        
    def call(self, x):
        
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embed_dim])
        
        encode_ind = self.get_code_indices(flattened)
        encodings = tf.one_hot(encode_ind, self.no_embeddings)
        quantized = tf.matmul(encodings, self.embeddings, transpose_b=True)
        
        quantized = tf.reshape(quantized, input_shape)
        
        comm_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        self.add_loss(self.beta * comm_loss + codebook_loss)
        
        quantized = x + tf.stop_gradient(quantized - x)
        
        return quantized
    
    def get_code_indices(self, flattened_inputs):
        
        sim = tf.matmul(flattened_inputs, self.embeddings)
        
        dists = (tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True) + 
                 tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * sim)
        
        encode_ind = tf.argmin(dists, axis=1)
        return encode_ind
        