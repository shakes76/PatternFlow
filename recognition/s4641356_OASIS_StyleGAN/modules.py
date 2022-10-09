import tensorflow as tf
import numpy as np

class adaIN(tf.keras.layers.Layer):
    """
    Custom Keras Neural Netork layer to conduct adaptive instance normalization
    As specified by StyleGAN. this is deterministic as the learnt scale and bias come from
    an externally trained dense layer.
    """
    def __init__(self, **kwargs) -> None:
        """
        Instantiate new adaIn layer
        """
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Called automatically on first 'call' passing the shape of input
        To allow weights to be allocated lazily
        Args:
            input (_type_): _description_ TODO
        """
        pass

    def call(self, input):
        """
        Preforms the layer's desired operation using trained weights

        Args:
            input (_type_): _description_ TODO [x (nxn),yscale (1x1),ybias (1x1)]
        """
        x,yscale,ybias = input
        mean = tf.math.reduce_mean(x)
        std = tf.math.reduce_std(x)
        return (yscale*mean) + ybias

class addNoise(tf.keras.layers.Layer):
    """
    Custom Keras Neural Network layer to add in specified noise scaled by a learnt factor
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Instantiate new addNoise layer
        """
        super().__init__(**kwargs)

    def build(self, input_shape):
        """
        Called automatically on first 'call' passing the shape of input
        To allow weights to be allocated lazily
        Args:
            input (_type_): _description_ TODO
        """
        #This layer has a single weight to train, the scaling of the noise
        self.noise_weight = self.add_weight(shape = [1], initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0), name = "noise_weight") #inherited from Layer

    def call(self, input):
        """
        Preforms the layer's desired operation using trained weights

        Args:
            input (_type_): _description_ TODO [x (nxn), noise (nxn)]
        """
        x,noise = input
        return x + (self.noise_weight*noise)

class StyleGANGenerator:
    pass

class StyleGANDiscriminator:
    pass

class StyleGAN(tf.keras.Model):
    pass