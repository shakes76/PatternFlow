"""
    Model module for AKOA StyleGAN.
    Defines the class StyleGan which contains 
    discriminator, generator, and other 
    relevant models.
    
    author: Richard Marron
    status: Development
"""

import tensorflow as tf
from tensorflow.keras import Sequential, layers

class StyleGan():
    def __init__(self):
        pass
    
    
    def mapping_network(self):
        """
        Create the mapping network which takes 
        a point from latent space and transforms
        it into a style vector.
        
        Return: Model
        """
        model = Sequential()
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(64, activation='relu'))
        
        return model
    
    
    def discriminator_model(self):
        """
        Define the discriminator model for the Style GAN
        
        Return: Discriminator Model
        """
        model = Sequential()
        #model.add(layers.Dense())
        pass
    
    def generator_model(self):
        """
        Define the generator model for the Style GAN
        
        Return: Generator Model
        """
        model = Sequential()
        pass
    
    def adaIN(self, x, y):
        """
        Calculate the adaIN value to be used in the model
        Params: 
            x : Vector which corresponds to feature map
            y : Tuple of vectors (y_s, y_b) called styles
            
        Return: AdaIN value (float)
        """
        y_s, y_b = y
        mu_x = tf.math.reduce_mean(x)
        sig_x = tf.math.reduce_std(x)
        return y_s * (x - mu_x)/sig_x + y_b