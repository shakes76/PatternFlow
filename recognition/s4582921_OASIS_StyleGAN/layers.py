"""
layers.py

File containing any modified layers in the StyleGAN2.

Requirements:
    - tensorflow-gpu - 2.4.1
    - matplotlib - 3.4.3

Author: Bobby Melhem
Python Version: 3.9.7

Code logic based from : https://github.com/manicman1999/StyleGAN2-Tensorflow-2.0/blob/989306792ca49dcbebb353c4f06c7b48aeb3a9e3/conv_mod.py
"""


import tensorflow as tf
from tensorflow.keras.layers import Conv2D, InputSpec
from tensorflow.keras.backend import expand_dims, sqrt, sum, square
from tensorflow.python.keras.utils import conv_utils

#Value for avoiding zero division error
DELTA = 0.000001

class Conv2DModulation(Conv2D):
    """
    Modified convolutional layer for applying weight emodulation and combining style and input layer.

    Attributes:
        demod : Whether to demodulate weights
        input_spec : specify input to be combination of layer and style
    """


    def __init__(self, filters, kernel_size, padding, demod=True):
        """Initialise the modified convolution layer inhereting from Conv2D"""

        super().__init__(filters=filters, kernel_size=kernel_size, padding=padding)
        self.demod = demod
        self.input_spec = [InputSpec(ndim = 4),
                            InputSpec(ndim = 2)]


    def build(self, input_shape):
        """
        Builds the kernel with specified input and filters

        Args:
            input_shape : shape of input
        """

        input_dimensions = input_shape[0][-1]
        kernel_shape = self.kernel_size + (input_dimensions, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      name='kernel',
                                      constraint=self.kernel_constraint)

        self.built = True


    def call(self, inputs):
        """
        Modified functionality of convolutional call.

        Args:
            inputs : inputs to apply call to
        Returns:
            The output layer with the applied convolution.
        """

        #Expand w to compatible shape with kernel
        expanded = expand_dims(expand_dims(expand_dims(inputs[1], axis=1), axis=1), axis=-1)

        kernel = expand_dims(self.kernel, axis=0)

        #Weight modulation
        weights = kernel * (expanded + 1)

        if self.demod:
            #Weight demodulation by normalising
            variance = sqrt(sum(square(weights), axis=[1,2,3], keepdims=True) + DELTA)
            weights = weights / variance

        #Fuse and apply kernels with appropriate matrix manipulation
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

