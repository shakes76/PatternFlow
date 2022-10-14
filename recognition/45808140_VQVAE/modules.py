import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

class VQ(layers.Layer):
    
    def __init__(self, no_embeddings, embed_dim, beta=0.25):
        super().__init__(name='VQ_layer')
        self.embed_dim = embed_dim
        self.no_embeddings = no_embeddings
        self.beta = beta
        
        w_init = tf.random_uniform_initializer()
        embed_shape = (self.embed_dim, self.no_embeddings)
        self.embeddings = tf.Variable(initial_value=w_init(embed_shape, dtype=tf.float32))
        
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
        
        dists = (tf.reduce_sum(flattened_inputs ** 2, axis=1, keepdims=True) 
                + tf.reduce_sum(self.embeddings ** 2, axis=0) 
                - 2 * sim)
        
        return tf.argmin(dists, axis=1)

class Encoder():

    def __init__(self, latent_dim=16, input_x=256):
        self.encoder = keras.Sequential(name='encoder')
        self.encoder.add(keras.Input(shape=(input_x, input_x, 1)))
        self.encoder.add(layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'))
        self.encoder.add(layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'))
        self.encoder.add(layers.Conv2D(latent_dim, 1, activation='relu', padding='same'))

    def get_encoder(self):
        return self.encoder

class Decoder():

    def __init__(self, latent_dim=16):
        self.decoder = keras.Sequential(name='decoder')
        self.decoder.add(keras.Input(shape=Encoder(latent_dim).get_encoder().output.shape[1:]))
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

class PixelCNNLayers(layers.Layer):
    
    def __init__(self, mask_type, **kwargs):
        super(PixelCNNLayers, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)
        
    def build(self, input_shape):
        self.conv.build(input_shape)
        
        kernel_shape = self.conv.kernel.get_shape()
        
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        
        if self.mask_type == 'B':
            self.mask[kernel_shapel[0] // 2, kernel_shape[1] // 2, ...] = 1.0
            
    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        
        return self.conv(inputs)

class ResidBlock(layers.Layer):
    
    def __init__(self, filters, **kwargs):
        super(ResidBlock, self).__init__(**kwargs)
        self.conv1 = keras.layers.Conv2D(filters=filters, kernel_size=1, activation='relu')
        
        self.pixel_conv = PixelCNNLayers(mask_type='B', filters=filters//2, kernel_size=3, 
                                         activation="relu", padding='same')
        self.conv2 = layers.Conv2D(filters=filters, kernel_size=1, activation='relu')
        
    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pixel_conv(x)
        x = self.conv2(x)
        return layers.add([inputs, x])

class PixelCNN():
    
    def __init__(self, input_shape, vq_trainer, no_resid=2, no_pixel_layers=2):
        pixel_inputs = keras.Input(shape=input_shape, dtype=tf.int32)
        one = tf.one_hot(pixel_inputs, vq_trainer.no_embeddings)
        
        x = PixelCNNLayers(mask_type='A', filters=128, kernel_size=7, activation='relu', padding='same')(one)
        
        for _ in range(no_resid):
            x = ResidBlock(filters=128)(x)
        for _ in range(no_pixel_layers):
            x = PixelCNNLayers(mask_type='B', filters=128, kernel_size=1, strides=1, 
                               activation='relu', padding='valid')(x)
            
        out = layers.Conv2D(filters=vq_trainer.no_embeddings, kernel_size=1, strides=1, padding='valid')(x)
        self.pixel_cnn = keras.Model(pixel_inputs, out, name='pixelCNN')
        
    def get_model(self):
        return self.pixel_cnn