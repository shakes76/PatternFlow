from tensorflow.keras import layers
import math
import tensorflow as tf

"""
The fourier encoding part of the perceiver transformer model
"""
class FourierEncode(layers.Layer)
    """
    Initialise the Encoding class
    Parameters:
        max_freq: type int, the maximum frequency of the fourier feature (by default parameter follows the paper spec)
        num_bands: type int, the number of frequency bands in fourier features
    """
    def __init__(self, max_freq=10, num_bands=4):
        super(FourierEncode, self).__init__()
        self.max_freq = max_freq
        self.num_bands = num_bands


    """
    Handles a call to the class with the image data given. 
    Parameters:
        data: An array containing the image data 
    
    Returns the image encoded with its fourier features
    """
    def call(self, data):
        # Here referencing the fourier encode code used from https://github.com/Rishit-dagli/Perceiver/tree/main/perceiver

        batch_size, *axis, _ = data.shape 
        axis = tuple(axis)

        rows, cols = axis[0], axis[1]
        # scales positions to [-1,1] and stack it, with the shape being list(tensor, tensor)
        # Here, shape = (*axis, 2)
        pos = tf.stack(tf.meshgrid(*list(map(lambda size: tf.linspace(-1.0, 1.0, num=size), axis)), indexing="ij"), axis=-1)

        # Here we obtain the encoded fourier features
        encoded_pos = self._fourier_encode(pos)
        del pos
        encoded_pos = tf.reshape(encoded_pos, (1, rows, cols, 2*(2*self.num_bands+1)))

        # repeat the process for batch_size times
        enc_pos = tf.repeat(enc_pos, repeats=batch_size, axis=0)
        # combine image with encoded position
        img_encode = tf.concat((imgs, enc_pos), axis=-1)

        # flatten image and return it 
        return tf.reshape(img_encode, (batch_size, rows*cols, -1))

    
    """
    Computes the Fourier features and concatenate it with the original position labels
    Params:
        pos: a tf matrix with positions scaled to range (-1, 1)
    Returns: matrix which contains concatenation of fourier feature and original position 
    """
    def _fourier_encode(self, pos):
        # modify the shape to be (*axis, 2 , 1)
        pos = tf.expand_dims(pos, -1)
        pos = tf.cast(pos, dtype=tf.float32)
        orig_pos = pos
        
        fourier_k = tf.experimental.numpy.logspace(
            start=0.0,
            stop=math.log(self.max_freq / 2) / math.log(10),
            num=self.num_bands,
            dtype=tf.float32,
        )
        # reshape to match the 4D position matrix 
        fourier_k = tf.reshape(fourier_k, (*((1,) * (len(pos.shape) - 1)), self.num_bands))

        # update the position to be fourier key * pi * pos
        pos = pos * fourier_k * math.pi

        # get fourier features: [sin(fkπxd), cos(fkπxd)]
        pos = tf.concat([tf.math.sin(pos), tf.math.cos(pos)], axis=-1)
        pos = tf.concat((pos, orig_pos), axis=-1)
        return pos
