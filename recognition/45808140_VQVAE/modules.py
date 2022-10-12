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
        
class Encoder():

    def __init__(self, latent_dim=16):
        self.encoder = keras.Sequential()
        self.encoder.add(keras.Input(shape=(28,28,1)))
        self.encoder.add(layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'))
        self.encoder.add(layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'))
        self.encoder.add(layers.Conv2D(latent_dim, 1, activation='relu', padding='same'))

    def get_encoder(self):
        return self.encoder

class Decoder():

    def __init__(self, latent_dim=16):
        self.decoder = keras.Sequential()
        self.decoder.add(keras.Input(shape=Encoder(latent_dim).get_encoder().output_shape[1:]))
        self.decoder.add(layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'))
        self.decoder.add(layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'))
        self.decoder.add(layers.Conv2DTranspose(1, 3, activation='relu', padding='same'))

    def get_decoder(self):
        return self.decoder

class VQVAE():
    
    def __init__(self, latent_dim=16, no_embeddings=64):
    
        self.vq = VQ(no_embeddings, latent_dim)
        self.encoder = get_encoder(latent_dim)
        self.decoder = get_decoder(latent_dim)
    
        self.inputs = keras.Input(shape=(28,28,1))
        encode_out = self.encoder(self.inputs)
        self.vq_latent_vecs = self.vq(encode_out)
        self.res = self.decoder(self.vq_latent_vecs)
    
        self.model = keras.Model(self.inputs, self.res, name='vqvae')
        
    def get_input_shape(self):
        return self.inputs
        
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_res(self):
        return self.res
        
    def get_model(self):
        return self.model