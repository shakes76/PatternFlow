from tensorflow.keras import layers
import math
import tensorflow as tf

MAX_FREQ = 10
NUM_BANDS = 6
LATENT_SIZE = 256
PROJ_SIZE = 27
LATENT_SHAPE = (LATENT_SIZE, PROJ_SIZE)
NUM_HEADS = 8

"""
    Applies fourier encoding to the image data given
    Returns the fourier encoding of the data
"""
def fourier_encode(data):
    # Here referencing the fourier encode code used from https://github.com/Rishit-dagli/Perceiver/tree/main/perceiver

    batch_size, *axis, _ = data.shape
    axis = tuple(axis)

    rows, cols = axis[0], axis[1]
    # scales positions to [-1,1] and stack it, with the shape being list(tensor, tensor)
    # Here, shape = (*axis, 2)
    pos = tf.stack(tf.meshgrid(*list(map(lambda size: tf.linspace(-1.0, 1.0, num=size), axis)), indexing="ij"),
                   axis=-1)
    pos = tf.cast(tf.expand_dims(pos, -1), dtype=tf.float32)

    # copy the array for encoding
    copy_pos = pos

    # Obtain the fourier encoding
    encoding = tf.reshape(tf.experimental.numpy.logspace(start=0.0, stop=math.log(MAX_FREQ / 2) / math.log(10)
                            , num=NUM_BANDS, dtype=tf.float32, ),
                    (*((1,) * (len(pos.shape) - 1)), NUM_BANDS))

    pos = copy_pos * encoding * math.pi
    # repeat the process for batch_size times
    parameterisation = [tf.math.sin(pos), tf.math.cos(pos)]
    # Repeat the Fourier Encoding for batch number of times and reshape it to the required dimension
    batch_size = data.shape[0]
    x_dim = data.shape[1]
    y_dim = data.shape[2]

    fourier_layer = tf.repeat(
        tf.reshape(tf.concat((tf.concat(parameterisation, axis=-1), copy_pos), axis=-1),
                   (1, x_dim, y_dim, PROJ_SIZE), repeats=batch_size, axis=0))


    # flatten image and return it
    return tf.reshape(tf.concat((data, fourier_layer), axis=-1), (batch_size, x_dim * y_dim, -1))
