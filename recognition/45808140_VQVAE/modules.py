import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp

"""Subclass of keras layers which defines the VQ layer of the VQVAE model"""
class VQ(layers.Layer):
    
    def __init__(self, no_embeddings, embed_dim, beta=0.25):
        super().__init__(name='VQ_layer')
        #latent dimensions
        self.embed_dim = embed_dim
        #number of embeddings/codebooks
        self.no_embeddings = no_embeddings
        #scale of reconstruction loss
        self.beta = beta
        
        #initialise random uniform priors
        w_init = tf.random_uniform_initializer()
        embed_shape = (self.embed_dim, self.no_embeddings)
        
        #Initialise embeddings to quantize
        self.embeddings = tf.Variable(initial_value=w_init(embed_shape, tf.float32))
        
    def call(self, x):

        input_shape = tf.shape(x)
        #flatten input to embedding dimensions
        flattened = tf.reshape(x, [-1, self.embed_dim])
        
        #get encodings and quantized latents and reshape to original input
        encode_ind = self.get_code_indices(flattened)
        encodings = tf.one_hot(encode_ind, self.no_embeddings)
        quantized = tf.reshape(tf.matmul(encodings, self.embeddings, transpose_b=True), 
            tf.shape(x))
        
        #calculate VQ loss to add to layer based on VQVAE paper
        comm_loss = tf.reduce_mean((tf.stop_gradient(quantized) - x) ** 2)
        codebook_loss = tf.reduce_mean((quantized - tf.stop_gradient(x)) ** 2)
        
        self.add_loss(self.beta * comm_loss + codebook_loss)
        
        #copy gradients from decoder input to encoder output (straight-through estimator)
        return x + tf.stop_gradient(quantized - x)
    
    def get_code_indices(self, flattened_in):
        """Calculate L2-norm distance between inputs and codebooks based on paper"""
        sim = tf.matmul(flattened_in, self.embeddings)
        
        dists = (tf.reduce_sum(flattened_in ** 2, axis=1, keepdims=True) 
                 + tf.reduce_sum(self.embeddings ** 2, axis=0) - 2 * sim)
        
        #take the minimum distance
        encode_ind = tf.argmin(dists, axis=1)
        return encode_ind 

"""Encoder model adapted from keras tutorial using Sequential model instead"""
class Encoder():

    def __init__(self, latent_dim=16, input_x=256):
        self.encoder = keras.Sequential(name='encoder')
        self.encoder.add(keras.Input(shape=(input_x, input_x, 1)))
        self.encoder.add(layers.Conv2D(32, 3, activation='relu', strides=2, padding='same'))
        self.encoder.add(layers.Conv2D(64, 3, activation='relu', strides=2, padding='same'))
        self.encoder.add(layers.Conv2D(latent_dim, 1, padding='same'))

    def get_encoder(self):
        return self.encoder
    
"""Decoder model adapted from keras tutorial using Sequential model instead"""
class Decoder():

    def __init__(self, latent_dim=16):
        self.decoder = keras.Sequential(name='decoder')
        self.decoder.add(keras.Input(shape=Encoder(latent_dim).get_encoder().output.shape[1:]))
        self.decoder.add(layers.Conv2DTranspose(64, 3, activation='relu', strides=2, padding='same'))
        self.decoder.add(layers.Conv2DTranspose(32, 3, activation='relu', strides=2, padding='same'))
        self.decoder.add(layers.Conv2DTranspose(1, 3, activation='relu', padding='same'))

    def get_decoder(self):
        return self.decoder

"""VQVAE model combining VQ, encoder and decoder models"""
class VQVAE_model(keras.Model):
    
    def __init__(self, input_x, train_var, latent_dim=16, no_embeddings=64, beta=0.25, **kwargs):
        super(VQVAE_model, self).__init__(**kwargs)
        #initialise variables
        self.latent_dim = latent_dim
        self.no_embeddings=no_embeddings
        self.train_var = train_var
        self.input_x = input_x
        
        #models
        self.encoder = Encoder(latent_dim=self.latent_dim, input_x=self.input_x).get_encoder()
        self.decoder = Decoder(latent_dim=self.latent_dim).get_decoder()
        self.vq = VQ(self.no_embeddings, self.latent_dim)
        
        #define loss functions, total_loss, reconstruction loss and VQ loss
        #calculated using loss functions in VQVAE paper
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
        """Get VQVAE model to demonstrate structure"""
        input_shape = keras.Input(shape=(self.input_x, self.input_x, 1))
        encode_out = self.encoder(input_shape)
        q_latents = self.vq(encode_out)
        recons = self.decoder(q_latents)
        
        return keras.Model(input_shape, recons, name='vqvae')
    
    def call(self, x):
        #forward pass of the model feeding through encoder then VQ then decoding
        x = self.encoder(x)
        x = self.vq(x)
        x = self.decoder(x)
        return x
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            #reconstructing images and calculating loss
            recons = self(x)
            recons_loss = tf.reduce_mean((x - recons) ** 2) / self.train_var
            total_loss = recons_loss + sum(self.vq.losses)
            
        #backpropagate gradients to train encoder
        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        #Update losses
        self.total_loss.update_state(total_loss)
        self.vq_loss.update_state(sum(self.vq.losses))
        self.recons_loss.update_state(recons_loss)
        
        return {'loss': self.total_loss.result(), 
                'vq_loss': self.vq_loss.result(),
                'reconstruction_loss': self.recons_loss.result()} 

    @property
    def metrics(self):
        return [self.total_loss, self.vq_loss, self.recons_loss]

"""Building Pixel Convolution layers"""
class PixelCNNLayers(layers.Layer):
    
    def __init__(self, mask_type, **kwargs):
        super(PixelCNNLayers, self).__init__()
        self.mask_type = mask_type
        self.conv = layers.Conv2D(**kwargs)
        
    def build(self, input_shape):
        #Initialise kernels
        self.conv.build(input_shape)

        #Create masks
        kernel_shape = self.conv.kernel.get_shape()
        self.mask = np.zeros(shape=kernel_shape)
        self.mask[: kernel_shape[0] // 2, ...] = 1.0
        self.mask[kernel_shape[0] // 2, : kernel_shape[1] // 2, ...] = 1.0
        
        if self.mask_type == 'B':
            self.mask[kernel_shape[0] // 2, kernel_shape[1] // 2, ...] = 1.0
            
    def call(self, inputs):
        self.conv.kernel.assign(self.conv.kernel * self.mask)
        return self.conv(inputs)

"""Building Residual blocks for PixelCNN"""
class ResidBlock(layers.Layer):
    
    def __init__(self, filters, **kwargs):
        #Build residual blocks using Pixel convolution layers
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

"""Complete PixelCNN model from Pixel convolution layers and residual blocks"""
class PixelCNN(keras.Model):
    
    def __init__(self, input_shape, vq_trainer, no_resid=2, no_pixel_layers=2):
        super(PixelCNN, self).__init__(name='PixelCNN')
        #Initialise shape based on number of embeddings
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
        
        #Define average loss
        self.total_loss = keras.metrics.Mean(name='total_loss')
        
    def get_model(self):
        return self.pixel_cnn
    
    def call(self, x):
        return self.pixel_cnn(x)
    
    def train_step(self, x):
        with tf.GradientTape() as tape:
            predictions = self(x)
            #Calculate loss
            loss = self.compiled_loss(x, predictions)
        
        #Backpropagate gradients
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        #Update loss
        self.total_loss.update_state(loss)
        return {'loss': self.total_loss.result()}