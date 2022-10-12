import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_probability as tfp
from modules import *

class VQVAE_train(keras.models.Model):
    
    def __init__(self, train_var, latent_dim=32, no_embeddings=128):
        super(VQVAE_train, self).__init__()
        self.train_var = train_var
        self.latent_dim = latent_dim
        self.no_embeddings = no_embeddings
        
        self.vqvae = VQVAE(self.latent_dim, self.no_embeddings).get_model()
        
        self.total_loss = keras.metrics.Mean(name='total_loss')
        self.recons_loss = keras.metrics.Mean(name='reconstruction_loss')
        self.vq_loss = keras.metrics.Mean(name='vq_loss')
        
    @property
    def metrics(self):
        return [self.total_loss, self.vq_loss, self.recons_loss]
    
    def train_step(self, x):
        
        with tf.GradientTape() as tape:
            recons = self.vqvae(x)
            
            recons_loss = (tf.reduce_mean((x - recons) ** 2) / self.train_var)
            total_loss = recons_loss + sum(self.vqvae.losses)
            
        grads = tape.gradient(total_loss, self.vqvae.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.vqvae.trainable_variables))
        
        self.total_loss.update_state(total_loss)
        self.vq_loss.update_state(sum(self.vqvae.losses))
        self.recons_loss.update_state(recons_loss)
        
        return {'loss': self.total_loss.result(), 
                'reconstruction_loss': self.recons_loss.result(), 
                'vq_loss': self.vq_loss.result()} 