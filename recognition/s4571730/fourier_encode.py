from tensorflow.keras import layers
import tensorflow as tf 
import math
class FourierEncode(layers.Layer):
    def __init__(self, max_freq=10, num_bands=4):
        super(FourierEncode, self).__init__()
        self.max_freq = max_freq
        self.num_bands = num_bands

    def call(self, patches):
        # TODO: fourier algo, combining encoded with data
        pass


def fourier_encode(x, max_freq=10, num_bands=4):
    rows = x.shape[0]
    cols = x.shape[1]
    xr = tf.linspace(-1, 1, rows)
    xc = tf.linspace(-1, 1, cols)
    xd = tf.reshape(tf.stack(tf.reverse(tf.meshgrid(xr, xc), axis=[-3]),axis=2), (rows, cols, 2))
    xd = tf.repeat(tf.expand_dims(xd, -1), repeats=[2*bands + 1], axis=3)

    freq = tf.experimental.numpy.logspace()
    # x = tf.expand_dims(x, -1)
    # x = tf.cast(x, dtype=tf.float32)
    # orig_x = x

    # scales = tf.reshape(tf.experimental.numpy.logspace(
    #     1.0,
    #     tf.math.log(max_freq / 2) / math.log(10),
    #     num=num_bands,
    #     dtype=tf.float32,
    # ), (1,1,1,2 * max_freq - 1) )
    # scales *= math.pi
    # x = x * scales 
    # return tf.concat((tf.concat([tf.math.sin(x), tf.math.cos(x)], axis=-1), orig_x), axis=-1)
    # return x



    

# def fourier_transform(img, bands, sampling_rate):
#     # data has 2 dimensions 
#     num_row, num_col, _ = img.shape
#     encodings = []
#     x_row = [(idx // num_col)/ (num_row - 1) * 2 - 1 for idx in list(range(num_row*num_col))] # row, col in range -1 1
#     x_col = [(idx % num_col)/ (num_col - 1) * 2 - 1 for idx in list(range(num_row*num_col))]
#     for input in range(num_col*num_row):
#         encoding = []
#         for xd in [x_row[input], x_col[input]]:
#             freq = np.logspace(0.0, math.log(sampling_rate/2) / math.log(10), bands, dtype=np.float32)
#             encoded_concat = []
#             for i in range(bands):
#                 encoded_concat.append(math.sin(freq[i] * math.pi * xd))
#                 encoded_concat.append(math.cos(freq[i] * math.pi * xd))
#             encoded_concat.append(xd)
#             encoding.extend(encoded_concat)
#         encodings.append(encoding)
#     return encodings
