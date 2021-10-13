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
    def __init__(self, input_shape: tuple, noise_dim: int):
        self.map_net = self.create_mapping_network()
        self.generator = self.create_generator(input_shape, noise_dim)
        self.discriminator = self.create_discriminator(input_shape)
        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    def get_discriminator_loss(self, real_output, fake_output):
        """
        Calculate the loss of the discriminator model.
        Params:
            real_output : Discriminator's results on real images
            fake_output : Discriminator's results on fake images
        """
        real_loss = self.cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def get_generator_loss(self, fake_output):
        """
        Calculate the loss of the generator model.
        Params:
            fake_output : Generator's results on fake images
        """
        return self.cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def create_mapping_network(self):
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
        
        self.map_net = model
        
        return model
    
    
    def create_discriminator(self, input_shape: tuple):
        """
        Define the discriminator model for the Style GAN
        
        Return: Discriminator Model
        """
        model = Sequential()
        model.add(layers.Conv3D(64, 3, strides=(2, 2, 2), padding='same',
                                input_shape=input_shape))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Conv3D(128, 3, strides=(2, 2, 2), padding='same'))
        model.add(layers.LeakyReLU())
        model.add(layers.Dropout(0.3))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        self.discriminator = model
        
        return model
    
    def create_generator(self, input_shape:tuple, noise_dim: int):
        """
        Define the generator model for the Style GAN
        
        Return: Generator Model
        """
        h, w, d = input_shape
        
        model = Sequential() 
        model.add(layers.Dense(240*180*3, use_bias=False, input_shape=(noise_dim,)))
        # model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
    
        model.add(layers.Reshape((15, 5, 1728, 2))) 
        assert model.output_shape == (None, 15, 5, 1728, 2) # 129600
    
        model.add(layers.Conv3DTranspose(288, 3, strides=(2, 3), padding='same', 
                                        use_bias=False))
        assert model.output_shape == (None, 30, 15, 288, 2) # 129600
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv3DTranspose(72, 3, strides=(2, 2), padding='same', 
                                        use_bias=False))
        assert model.output_shape == (None, 60, 30, 72, 2) # 129600
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        
        model.add(layers.Conv2DTranspose(18, 3, strides=(2, 2), padding='same', 
                                        use_bias=False))
        assert model.output_shape == (None, 120, 60, 18, 2)
        model.add(layers.BatchNormalization()) # 129600
        model.add(layers.LeakyReLU())
    
        # 240x180
        model.add(layers.Conv2DTranspose(3, 3, strides=(2, 3), padding='same', 
                                        use_bias=False, activation='tanh'))
        assert model.output_shape == (None, h, w, d, 1) # 129600
        
        self.generator = model
        
        return model
    
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