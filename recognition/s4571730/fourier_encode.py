from tensorflow.keras import layers
import tensorflow as tf 
import math

"""
Fourier class to embed image data with fourier encoding, as described in the paper
"""
class FourierEncode(layers.Layer):
    """
    Init the Fourier class

    Params:
        max_freq: int, Nyquist frequency of the Fourier features
        num_bands: int, number of frequency bands in fourier features
    """
    def __init__(self, max_freq=10, num_bands=4):
        super(FourierEncode, self).__init__()
        self.max_freq = max_freq
        self.num_bands = num_bands

    """
    Process a call to the class with supplied data

    Params:
        imgs: an array containing img data

    Returns:
        img_encode: img encoded with its fourier features
    """
    def call(self, imgs):
        # Based on fourier encode from https://github.com/Rishit-dagli/Perceiver/blob/main/perceiver/
        batch_size, *axis, _ = imgs.shape
        axis = tuple(axis)
        rows, cols = axis[0], axis[1]
        # scales positions to [-1,1] and stack it
        # shape = list(tensor, tensor)
        axis_pos = list(map(lambda size: tf.linspace(-1.0, 1.0, num=size), axis))
        # shape = (*axis, 2)
        pos = tf.stack(tf.meshgrid(*axis_pos, indexing="ij"), axis=-1)

        # get the encoded fourier features
        enc_pos = self._fourier_encode(pos)
        del pos
        enc_pos = tf.reshape(enc_pos, (1, rows, cols, 2*(2*self.num_bands+1)))

        # repeat batch_size times
        enc_pos = tf.repeat(enc_pos, repeats=batch_size, axis=0)
        # combine image with encoded position
        img_encode = tf.concat((imgs, enc_pos), axis=-1)

        # flatten image
        img_encode = tf.reshape(img_encode, (batch_size, rows*cols, -1)) 
        return img_encode

    """
    Calculate the Fourier features and concat it into original position labels

    Params:
        pos: a tf matrix containing the positions scaled to range (-1, 1)

    Returns:
        concatenation of fourier feature and original position matrix
    """
    def _fourier_encode(self, pos):
        # shape = (*axis, 2 , 1)
        pos = tf.expand_dims(pos, -1)
        pos = tf.cast(pos, dtype=tf.float32)
        orig_pos = pos
        
        fk = tf.experimental.numpy.logspace(
            start=0.0,
            stop=math.log(self.max_freq / 2) / math.log(10),
            num=self.num_bands,
            dtype=tf.float32,
        )
        # reshape to match position matrix (4D)
        fk = tf.reshape(fk, (*((1,) * (len(pos.shape) - 1)), self.num_bands))

        # get fkπxd
        pos = pos * fk * math.pi

        # get fourier features: [sin(fkπxd), cos(fkπxd)]
        pos = tf.concat([tf.math.sin(pos), tf.math.cos(pos)], axis=-1)
        pos = tf.concat((pos, orig_pos), axis=-1)
        return pos