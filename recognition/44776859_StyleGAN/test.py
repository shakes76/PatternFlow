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
from tensorflow.keras.layers import Layer, Input, Add, Multiply, Dense, Flatten, Reshape, LeakyReLU, Conv2D, \
    Conv2DTranspose, UpSampling2D
from tensorflow.keras import backend as K

import numpy as np
from matplotlib import pyplot as plt


# IMAGE_WIDTH = 28
# START_SIZE = (7,7,IMAGE_WIDTH)

# Define constants.
BATCH_SIZE = 2
IMAGE_SIZE = 28
IMAGE_DIMS = (IMAGE_SIZE, IMAGE_SIZE)
NUM_CHANNELS = 1

START_DIMS = (7,7)


x = np.array([[[[5,5]]]])
up = UpSampling2D()
x2 = up(x)


# Input b and g should be 1x1xC
class AdaInstanceNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 **kwargs):
        super(AdaInstanceNormalization, self).__init__(**kwargs)
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale

    def build(self, input_shape):

        dim = input_shape[0][self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape[0]) + '.')

        super(AdaInstanceNormalization, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        input_shape = K.int_shape(inputs[0])
        reduction_axes = list(range(0, len(input_shape)))

        beta = tf.reshape(inputs[1], shape=(input_shape[0], 1, 1, input_shape[-1]))
        gamma = tf.reshape(inputs[2], shape=(input_shape[0], 1, 1, input_shape[-1]))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]
        mean = K.mean(inputs[0], reduction_axes, keepdims=True)
        stddev = K.std(inputs[0], reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs[0] - mean) / stddev

        return normed * gamma + beta

    # def get_config(self):
    #     config = {
    #         'axis': self.axis,
    #         'momentum': self.momentum,
    #         'epsilon': self.epsilon,
    #         'center': self.center,
    #         'scale': self.scale
    #     }
    #     base_config = super(AdaInstanceNormalization, self).get_config()
    #     return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]


# class ScalarMult(tf.keras.layers.Layer):
#     def __init__(self):
#         super(ScalarMult, self).__init__()
#
#     def call(self, inputs, *args, **kwargs):
#         scalar, x = inputs
#         # const = tf.constant([2.], shape=(1,1))
#         return scalar * x
#         # return tf.math.scalar_mul(tf.reshape(scalar, shape=(-1)), x)

#
# class ConstLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(ConstLayer, self).__init__()
#
#     def call(self, inputs, *args, **kwargs):
#         return tf.ones(shape=((BATCH_SIZE,) + START_DIMS + (IMAGE_SIZE,)))


# class ConstLayer(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(ConstLayer, self).__init__(**kwargs)
#
#     # def build(self, input_shape):
#     #     super(ConstLayer, self).build(input_shape)
#     #     self.shape = input_shape
#
#     def call(self, inputs, *args, **kwargs):
#         input_shape = K.int_shape(inputs)
#         return tf.ones(shape=((input_shape[0],) + START_DIMS + (IMAGE_SIZE,)))
#         # return tf.ones(shape=((self.shape[0],) + START_DIMS + (IMAGE_SIZE,)))

class ConstOnes(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConstOnes, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConstOnes, self).build(input_shape)
        output_dim = input_shape[-1]
        self.kernel = self.add_weight(
            shape=(output_dim, output_dim),
            trainable=False
        )

    def call(self, inputs, *args, **kwargs):
        input_shape = K.int_shape(inputs)
        print(input_shape)
        # if input_shape[0] is not None:

        # sh = tf.shape(input_shape)
        # print(sh)
        # x = tf.ones(shape=((tf.shape(input_shape)[0],) + START_DIMS + (IMAGE_SIZE,)))
        # print(x)
        # return tf.ones(shape=((tf.shape(input_shape)[0],) + START_DIMS + (IMAGE_SIZE,)))

        print(inputs.get_shape())
        print(inputs.get_shape()[0])
        ones = tf.ones(shape=input_shape[-1])
        print(ones)
        # x = inputs[None, :, :, :]
        # x = tf.reshape(inputs, shape=(None, 7,7,28))
        return ones

    def compute_output_shape(self, input_shape):
        return input_shape[0]

# class SynthBlock(tf.keras.layers.Layer):
#     def __init__(self):
#         super(SynthBlock, self).__init__()
#
#     def call(self, inputs, *args, **kwargs):
#         const = tf.ones(shape=(1, 4, 4, 512))
#         return Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2),
#                                padding='same')(const)


def generate_latent_coordinates(batches, dimensions):
    return tf.random.normal([batches, dimensions])


inputs = generate_latent_coordinates(BATCH_SIZE, IMAGE_SIZE)

# Mapping network
dense1 = Dense(IMAGE_SIZE, activation='linear')(inputs)
dense2 = Dense(IMAGE_SIZE, activation='linear')(dense1)
dense3 = Dense(IMAGE_SIZE, activation='linear')(dense2)

# -------------------------------------------------------------------------------------------------

# const = ConstLayer()(inputs)
const = ConstOnes(input_shape=(IMAGE_SIZE,), name='Constant')
const = const(inputs)

a1scale = Dense(IMAGE_SIZE, activation='linear')(dense3)
a1bias = Dense(IMAGE_SIZE, activation='linear')(dense3)

ada1 = AdaInstanceNormalization(input_shape=(7, 7, IMAGE_SIZE))([const, a1bias, a1scale])

conv1 = Conv2DTranspose(filters=IMAGE_SIZE, kernel_size=(3, 3), strides=(1,1), padding='same')(ada1)
act1 = LeakyReLU(alpha=0.2)(conv1)

ada2 = AdaInstanceNormalization(input_shape=(7,7, IMAGE_SIZE))([act1, a1bias, a1scale])

# -------------------------------------------------------------------------------------------------

conv2 = Conv2DTranspose(filters=IMAGE_SIZE, kernel_size=(3, 3), strides=(2, 2), padding='same')(ada2)
act2 = LeakyReLU(alpha=0.2)(conv2)

a2scale = Dense(IMAGE_SIZE, activation='linear')(dense3)
a2bias = Dense(IMAGE_SIZE, activation='linear')(dense3)

ada3 = AdaInstanceNormalization(input_shape=(14,14, IMAGE_SIZE))([act2, a2bias, a2scale])

conv3 = Conv2DTranspose(filters=IMAGE_SIZE, kernel_size=(3, 3), strides=(2, 2), padding='same')(ada3)
act3 = LeakyReLU(alpha=0.2)(conv3)

ada4 = AdaInstanceNormalization(input_shape=(28,28, IMAGE_SIZE))([act3, a2bias, a2scale])

# -------------------------------------------------------------------------------------------------

conv4 = Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh')(ada4)

plt.imshow(conv4[0], cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()











# Attempt 3, working

# z = generate_latent_coordinates(batches=1, dimensions=512)
#
# # Mapping network
# dense1 = Dense(512, activation='linear')
# dense1_out = dense1(z)
# dense2 = Dense(512, activation='linear')
# dense2_out = dense2(dense1_out)
# dense3 = Dense(512, activation='linear')
# dense3_out = dense3(dense2_out)
#
# # -------------------------------------------------------------------------------------------------
#
# const = ConstLayer()(z)
#
# a1scale = Dense(512, activation='linear')(dense3_out)
# a1bias = Dense(512, activation='linear')(dense3_out)
#
# ada1 = AdaInstanceNormalization(input_shape=(4, 4, 512))([const, a1bias, a1scale])
#
# conv1 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(ada1)
# act1 = LeakyReLU(alpha=0.2)(conv1)
#
# ada2 = AdaInstanceNormalization(input_shape=(8, 8, 512))([act1, a1bias, a1scale])
#
# # -------------------------------------------------------------------------------------------------
#
# conv2 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(ada2)
# act2 = LeakyReLU(alpha=0.2)(conv2)
#
# a2scale = Dense(512, activation='linear')(dense3_out)
# a2bias = Dense(512, activation='linear')(dense3_out)
#
# ada3 = AdaInstanceNormalization(input_shape=(16, 16, 512))([act2, a2bias, a2scale])
#
# conv3 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')(ada3)
# act3 = LeakyReLU(alpha=0.2)(conv3)
#
# ada4 = AdaInstanceNormalization(input_shape=(32, 32, 512))([act3, a2bias, a2scale])
#
# # -------------------------------------------------------------------------------------------------
#
# conv4 = Conv2DTranspose(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='tanh')(ada4)
#
# plt.imshow(conv4[0], cmap='gray')
# plt.axis('off')
# plt.tight_layout()
# plt.show()




# Attempt 1, working

# class ScalarMult(tf.keras.layers.Layer):
#     def __init__(self):
#         super(ScalarMult, self).__init__()
#
#     def call(self, inputs, *args, **kwargs):
#         scalar, x = inputs
#         # const = tf.constant([2.], shape=(1,1))
#         return scalar * x
#         # return tf.math.scalar_mul(tf.reshape(scalar, shape=(-1)), x)
#
#
# class ConstLayer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(ConstLayer, self).__init__()
#
#     def call(self, inputs, *args, **kwargs):
#         return tf.ones(shape=(1, 4, 4, 512))
#
#
# class SynthBlock(tf.keras.layers.Layer):
#     def __init__(self):
#         super(SynthBlock, self).__init__()
#
#     def call(self, inputs, *args, **kwargs):
#         const = tf.ones(shape=(1, 4, 4, 512))
#         return Conv2DTranspose(filters=128, kernel_size=(3, 3), strides=(2, 2),
#                                padding='same')(const)
#
#
# def generate_latent_coordinates(batches, dimensions):
#     return tf.random.normal([batches, dimensions])
#
#
# array = np.ones(shape=(1, 512))
#
# const1 = ConstLayer()
# const = const1(array)
#
# conv1 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')
# conv1_out = conv1(const)
# act1 = LeakyReLU(alpha=0.2)
# act1_out = act1(conv1_out)
#
# dense1 = Dense(1, activation='linear')
# dense1_out = dense1(array)
#
# mult1 = ScalarMult()
# mult1_out = mult1((dense1_out, act1_out))
#
# conv2 = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='tanh')
# conv2_out = conv2(mult1_out)
#
#
# plt.imshow(conv2_out[0], cmap='gray')
# plt.axis('off')
# plt.tight_layout()
# plt.show()


# Attempt 2, working

# z = generate_latent_coordinates(batches=1, dimensions=512)
#
# const1 = ConstLayer()
# const = const1(z)
#
# dense1 = Dense(512, activation='linear')
# dense1_out = dense1(z)
# dense2 = Dense(512, activation='linear')
# dense2_out = dense2(dense1_out)
# dense3 = Dense(512, activation='linear')
# dense3_out = dense3(dense2_out)
#
# conv1 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')
# conv1_out = conv1(const)
# act1 = LeakyReLU(alpha=0.2)
# act1_out = act1(conv1_out)
#
# # Should be 512*2 neurons for A, as we need 512 ys and 512 yb, one ys/yb for each conv dimension.
# a1 = Dense(1, activation='linear')
# a1_out = a1(dense3_out)
#
# mult1 = ScalarMult()
# mult1_out = mult1((a1_out, act1_out))
#
# conv2 = Conv2DTranspose(filters=512, kernel_size=(3, 3), strides=(2, 2), padding='same')
# conv2_out = conv2(mult1_out)
# act2 = LeakyReLU(alpha=0.2)
# act2_out = act2(conv2_out)
#
# a2 = Dense(1, activation='linear')
# a2_out = a2(dense3_out)
#
# mult2 = ScalarMult()
# mult2_out = mult1((a2_out, act2_out))
#
# conv3 = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='tanh')
# conv3_out = conv3(mult2_out)
#
# plt.imshow(conv3_out[0], cmap='gray')
# plt.axis('off')
# plt.tight_layout()
# plt.show()
