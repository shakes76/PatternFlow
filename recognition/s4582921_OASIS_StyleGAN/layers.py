import tensorflow as tf
from tensorflow.keras.layers import Conv2D, InputSpec
from tensorflow.keras.backend import expand_dims, sqrt, sum, square
from tensorflow.python.keras.utils import conv_utils

DELTA = 0.000001

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

class Conv2DModulation(Conv2D):


    def __init__(self, filters, kernel_size, padding, demod=True):

        super().__init__(filters=filters, kernel_size=kernel_size, padding=padding)
        self.demod = demod
        self.input_spec = [InputSpec(ndim = 4),
                            InputSpec(ndim = 2)]


    def build(self, input_shape):

        input_dimensions = input_shape[0][-1]
        kernel_shape = self.kernel_size + (input_dimensions, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      name='kernel',
                                      constraint=self.kernel_constraint)

        self.built = True


    def call(self, inputs):

        expanded = expand_dims(expand_dims(expand_dims(inputs[1], axis=1), axis=1), axis=-1)

        kernel = expand_dims(self.kernel, axis=0)

        weights = kernel * (expanded + 1)

        if self.demod:
            variance = sqrt(sum(square(weights), axis=[1,2,3], keepdims=True) + DELTA)
            weights = weights / variance

        output = tf.transpose(inputs[0], [0, 3, 1, 2])
        output = tf.reshape(output, [1, -1, output.shape[2], output.shape[3]])

        w = tf.transpose(weights, [1, 2, 3, 0, 4])
        w = tf.reshape(w, [weights.shape[1], weights.shape[2], weights.shape[3], -1])

        output = tf.transpose(output, [0, 2, 3, 1])
        output = tf.nn.conv2d(output, w, strides=self.strides, padding="SAME", data_format="NHWC")
        output = tf.transpose(output, [0, 3, 1, 2])

        output = tf.reshape(output, [-1, self.filters, tf.shape(output)[2], tf.shape(output)[3]])
        output = tf.transpose(output, [0, 2, 3, 1])

        return output

