from tensorflow.keras import layers
import tensorflow as tf 
import math
# from einops import rearrange, repeat

class FourierEncode(layers.Layer):
    def __init__(self, max_freq=10, num_bands=4):
        super(FourierEncode, self).__init__()
        self.max_freq = max_freq
        self.num_bands = num_bands

    def call(self, imgs):
        # Based on fourier encode from https://github.com/Rishit-dagli/Perceiver/blob/main/perceiver/
        xxx, *axis, _ = imgs.shape
        batch_size = 32
        rows, cols = axis[0], axis[1]

        # scales positions to [-1,1] and stack it
        # shape = list(tensor, tensor)
        axis_pos = list(map(lambda size: tf.linspace(-1.0, 1.0, num=size), axis))
        # shape = (*axis, 2)
        pos = tf.stack(tf.meshgrid(*axis_pos, indexing="ij"), axis=-1)

        # get the encoded fourier features
        enc_pos = self._fourier_encode(pos)
        del pos
        # concat 
        # enc_pos = rearrange(enc_pos, "... n d -> ... (n d)")
        enc_pos = tf.reshape(enc_pos, (1, rows, cols, 2*(2*self.num_bands+1)))

        # repeat batch_size times
        # enc_pos = repeat(enc_pos, "... -> b ...", b=batch)
        enc_pos = tf.repeat(enc_pos, repeats=batch_size, axis=0)

        img_encode = tf.concat((imgs, enc_pos), axis=-1)

        # combine height and width
        # img_encode = rearrange(imgs, "b ... d -> b (...) d")
        img_encode = tf.reshape(img_encode, (batch_size, rows*cols, -1)) 
        return img_encode

    """
    Calculate the Fourier features and concat it into original position labels
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
        # scales = scales[(*((None,) * (len(pos.shape) - 1)), Ellipsis)]
        fk = tf.reshape(fk, (*((1,) * (len(pos.shape) - 1)), self.num_bands))

        # get fkπxd
        pos = pos * fk * math.pi

        # get fourier features: [sin(fkπxd), cos(fkπxd)]
        pos = tf.concat([tf.math.sin(pos), tf.math.cos(pos)], axis=-1)
        pos = tf.concat((pos, orig_pos), axis=-1)
        return pos


        # batch_size, rows, cols, dim = tf.shape(imgs)
        # x_row = tf.linspace(-1, 1, rows) 
        # x_col = tf.linspace(-1, 1, cols) 

        # # Create a grid of positional encodings
        # x_d = tf.reshape(tf.stack(tf.reverse(tf.meshgrid(x_row, x_col), axis=[-3]), axis=2), (rows,cols,2))
        # x_d = tf.expand_dims(x_d, -1)

        # # repeat it to size 2bands + 1
        # xd = tf.repeat(x_d, repeats=[2 * self.num_bands + 1], axis=3)   # (r, c, 2, 2bands + 1)

        # # log_scale
        # log_sample = tf.math.log(self.max_freq/2) / tf.math.log(10)

        # # get frequency fk
        # freqs = math.pi * tf.experimental.numpy.logspace(0, log_sample.numpy(), self.bands, dtype=tf.float32) 

        # # get fourier features: sin(fkπxd), cos(fkπxd)]
        # ff = tf.concat([tf.math.sin(freqs), tf.math.cos(freqs), tf.constant(1)], axis=0)
        # ff = tf.reshape(ff, shape=(1,1,1,2*self.bands+1))
        # ff = xd * tf.cast(ff, dtype=tf.double)
        
        # # shape = (batch, img_row, img_col, 2*(2*bands + 1))
        # ff = tf.reshape(ff, (1, rows, cols, 2*(2*self.bands+1)))
        # ff = tf.repeat(ff, repeats=[batch_size], axis=0)

        # # concat with data and flatten
        # img_encode = tf.concat((imgs, ff), axis=-1)
        # img_encode = tf.reshape(img_encode, (batch_size, rows*cols, -1)) 
        # return img_encode
