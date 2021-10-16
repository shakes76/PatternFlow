import os

# Suppress tensorflow logging:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf

print(f'Tensorflow version: {tf.__version__}')
print(f'Tensorflow CUDA {"is" if tf.test.is_built_with_cuda() else "is not"} available.')
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            print('Tensorflow set GPU memory growth to True.')
    except RuntimeError as e:
        print(e)
print(f'Tensorflow {"is" if tf.executing_eagerly() else "is not"} executing eagerly.')

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Add, Dense, Flatten, Reshape, LeakyReLU, Conv2D, Conv2DTranspose

import numpy as np


class StandardDeviation(tf.keras.layers.Layer):
    def __init__(self):
        super(StandardDeviation, self).__init__()

    def call(self, inputs, *args, **kwargs):
        std = tf.math.reduce_std(inputs)
        return tf.math.multiply(std, inputs)


array = np.random.random(size=(5, 5))
print(array.shape)

layer = StandardDeviation()
res = layer(array)

print(res)
