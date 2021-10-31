from tensorflow.keras import layers
import tensorflow as tf 
import math

class FourierEncode(layers.Layer):
    def __init__(self, max_freq=10, freq_ban=4):
        super(FourierEncode, self).__init__()
        self.max_freq = max_freq
        self.freq_ban = freq_ban

    def call(self, imgs):
        
        batch_size, *axis, _ = imgs.shape
        axis = tuple(axis)
        rows, cols = axis[0], axis[1]


        axis_pos = list(map(lambda size: tf.linspace(-1.0, 1.0, num=size), axis))

        pos = tf.stack(tf.meshgrid(*axis_pos, indexing="ij"), axis=-1)


        enc_pos = self._fourier_encode(pos)
        del pos
        enc_pos = tf.reshape(enc_pos, (1, rows, cols, 2*(2*self.freq_ban+1)))


        enc_pos = tf.repeat(enc_pos, repeats=batch_size, axis=0)

        img_encode = tf.concat((imgs, enc_pos), axis=-1)


        img_encode = tf.reshape(img_encode, (batch_size, rows*cols, -1)) 
        return img_encode




    def _fourier_encode(self, pos):

        pos = tf.expand_dims(pos, -1)
        pos = tf.cast(pos, dtype=tf.float32)
        orig_pos = pos
        
        fk = tf.experimental.numpy.logspace(
            start=0.0,
            stop=math.log(self.max_freq / 2) / math.log(10),
            num=self.freq_ban,
            dtype=tf.float32,
        )

        fk = tf.reshape(fk, (*((1,) * (len(pos.shape) - 1)), self.freq_ban))


        pos = pos * fk * math.pi


        pos = tf.concat([tf.math.sin(pos), tf.math.cos(pos)], axis=-1)
        pos = tf.concat((pos, orig_pos), axis=-1)
        return pos