from tensorflow.keras import layers
import tensorflow as tf 
import math
class FourierEncode(layers.Layer):
    def __init__(self, max_freq=10, num_bands=4):
        super(FourierEncode, self).__init__()
        self.max_freq = max_freq
        self.num_bands = num_bands

    def call(self, imgs):
        batch_size, rows, cols, dim = tf.shape(imgs)
        x_row = tf.linspace(-1, 1, rows) 
        x_col = tf.linspace(-1, 1, cols) 

        # Create a grid of positional encodings
        x_d = tf.reshape(tf.stack(tf.reverse(tf.meshgrid(x_row, x_col), axis=[-3]), axis=2), (rows,cols,2))
        x_d = tf.expand_dims(x_d, -1)

        # repeat it to size 2bands + 1
        xd = tf.repeat(x_d, repeats=[2 * self.num_bands + 1], axis=3)   # (r, c, 2, 2bands + 1)

        # log_scale
        log_sample = tf.math.log(self.max_freq/2) / tf.math.log(10)

        # get frequency fk
        freqs = math.pi * tf.experimental.numpy.logspace(0, log_sample.numpy(), self.bands, dtype=tf.float32) 

        # get fourier features: sin(fkπxd), cos(fkπxd)]
        ff = xd * tf.cast(tf.reshape(tf.concat([tf.math.sin(freqs), tf.math.cos(freqs), tf.constant(1)], axis=0), 
                    shape=(1,1,1,2*self.bands+1)), dtype=tf.double)

        # shape = (batch, img_row, img_col, 2*(2*bands + 1))
        ff = tf.repeat(tf.reshape(ff, (1, rows, cols, 2*(2*self.bands+1))), repeats=[batch_size], axis=0)
        # concat with data and flatten
        ff = tf.reshape(tf.concat((imgs, ff), axis=-1), (batch_size, rows*cols, -1)) 
        return ff