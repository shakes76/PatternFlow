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

    def build(self, input_shape: np.array) -> None:
        """
        Called automatically on first 'call' passing the shape of input
        To allow weights to be allocated lazily

        Args:
            input (np.array): Shape of input passed to layer
        """
        pass

    def call(self, input: list[tf.Tensor,tf.Tensor]) -> tf.Tensor:
        """
        Performs deterministic adaIN

        Args:
            input (list[tf.Tensor,tf.Tensor]): list of tensors, the first is the working image layer, 
                    the second is a (,2) tensor containing the corespondingfeature scale and bias

        Raises:
            ValueError: If the second input tensor is not exactly two values

        Returns:
            tf.Tensor: image layer scaled and biased (same dimensions as first input tensor)
        """

        if not (input[1].shape[1] == 2):
            raise ValueError("Second input must be of shape (,2), recieved {}".format(input[1].shape))

        x,y = input
        yscale,ybias = tf.split(y,2,axis = 1)#axes shifted by 1 to account for batches
        mean = tf.math.reduce_mean(x)
        std = tf.math.reduce_std(x)
        return (yscale[1:0]*(x-mean)/std) + ybias[1:0]

class addNoise(tf.keras.layers.Layer):
    """
    Custom Keras Neural Network layer to add in specified noise scaled by a learnt factor
    """
    
    def __init__(self, **kwargs) -> None:
        """
        Instantiate new addNoise layer
        """
        super().__init__(**kwargs)

    def build(self, input_shape: np.array) -> None:
        """
        Called automatically on first 'call' passing the shape of input
        To allow weights to be allocated lazily

        Args:
            input (np.array): Shape of input passed to layer
        """
        #This layer has a single weight to train, the scaling of the noise
        self.noise_weight = self.add_weight(shape = [1], initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=1.0), name = "noise_weight") #inherited from Layer

    def call(self, input: list[tf.Tensor,tf.Tensor]) -> tf.Tensor:
        """
        Preforms the layer's desired operation using trained weights

        Args:
            input (list[tf.Tensor,tf.Tensor]): List of two Tensors, the first is the current working image layer, and the second is the corresponding matrix of noise to add to it. 
                    The two tensors must have matching dimensions

        Raises:
            ValueError: If the two input tensors do not have matching dimension

        Returns:
            tf.Tensor: The image tensor with the noise scaled by learnt weight added to it
        """
        if not (input[0].shape == input[1].shape):
            raise ValueError("Inputs must be of same shape, recieved the following: Input 1: {}, Input 2: {}".format(input[0].shape,input[1].shape))

        x,noise = input
        return x + (self.noise_weight*noise)

class StyleGAN(tf.keras.Model):
    pass