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


inputs = Input(shape=(9))
reshape1 = Reshape(target_shape=(3,3))(inputs)

model = Model(inputs=inputs, outputs=reshape1)

ones = np.ones(shape=(1, 9))
res = model(ones, training=False)
