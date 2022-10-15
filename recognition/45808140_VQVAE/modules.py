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
        
        self.embeddings = tf.Variable(initial_value=w_init(embed_shape, tf.float32))
        
    def call(self, x):
        
        input_shape = tf.shape(x)
        flattened = tf.reshape(x, [-1, self.embed_dim])
        
        encode_ind = self.get_code_indices(flattened)
        encodings = tf.one_hot(encode_ind, self.no_embeddings)
        quantized = tf.reshape(tf.matmul(encodings, self.embeddings, transpose_b=True), tf.shape(x))
        
        comm_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        
        self.add_loss(self.beta * comm_loss + codebook_loss)
        
        return x + tf.stop_gradient(quantized - x)
    
    def get_code_indices(self, flattened_in):
        
        sim = tf.matmul(flattened_in, self.embeddings)
        
        dists = (tf.reduce_sum(flattened_in ** 2, axis=1, keepdims=True) 
                 + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * sim)
        
        encode_ind = tf.argmin(dists, axis=1)
        return encode_ind 

class Encoder():

    def __init__(self, latent_dim=16, input_x=256):
        self.encoder = keras.Sequential(name='encoder')
        self.encoder.add(keras.Input(shape=(input_x, input_x, 1)))
        self.encoder.add(layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'))
        self.encoder.add(layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'))
        self.encoder.add(layers.Conv2D(latent_dim, 1, padding='same'))

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

class VQVAE_model(keras.Model):
    
    def __init__(self, input_x, train_var, latent_dim=16, no_embeddings=64, beta=0.25, **kwargs):
        super(VQVAE_model, self).__init__(**kwargs)
        
        self.latent_dim = latent_dim
        self.no_embeddings=no_embeddings
        self.train_var = train_var
        self.input_x = input_x
        
        self.encoder = Encoder().get_encoder()
        self.decoder = Decoder().get_decoder()
        self.vq = VQ(self.no_embeddings, self.latent_dim)
        
        self.total_loss = keras.metrics.Mean(name='total_loss')
        self.recons_loss = keras.metrics.Mean(name='reconstruction_loss')
        self.vq_loss = keras.metrics.Mean(name='vq_loss')
        
    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder
    
    def get_vq(self):
        return self.vq
    
    def get_model(self):
        input_shape = keras.Input(shape=(self.input_x, self.input_x, 1))
        encode_out = self.encoder(input_shape)
        q_latents = self.vq(encode_out)
        recons = self.decoder(q_latents)
        
        return keras.Model(input_shape, recons, name='vqvae')
    
    def call(self, x):
        x = self.encoder(x)
        x = self.vq(x)
        x = self.decoder(x)
        return x
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            recons = self(x)
            recons_loss = tf.reduce_mean((x - recons) ** 2) / self.train_var
            total_loss = recons_loss + sum(self.vq.losses)
            
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        self.total_loss.update_state(total_loss)
        self.vq_loss.update_state(sum(self.vq.losses))
        self.recons_loss.update_state(recons_loss)
        
        return {'loss': self.total_loss.result(), 
                'vq_loss': self.vq_loss.result(),
                'reconstruction_loss': self.recons_loss.result()} 
    @property
    def metrics(self):
        return [self.total_loss, self.vq_loss, self.recons_loss]
        
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
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0
            
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

class PixelCNN(keras.Model):
    
    def __init__(self, input_shape, vq_trainer, no_resid=2, no_pixel_layers=2):
        super(PixelCNN, self).__init__(name='PixelCNN')
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
        
        self.total_loss = keras.metrics.Mean(name='total_loss')
        
    def get_model(self):
        return self.pixel_cnn
    
    def call(self, x):
        return self.pixel_cnn(x)
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            predictions = self(x)
            
            loss = self.compiled_loss(x, predictions)
            
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        self.total_loss.update_state(loss)
        return {'loss': self.total_loss.result()}