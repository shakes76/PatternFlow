# !/user/bin/env python
"""
The generator and discriminator models of the StyleGAN
"""

from math import log2
import numpy as np
import tensorflow as tf
from tensorflow.keras import initializers, layers, Model, constraints

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"


def _get_layers(init_resolution, final_resolution):
    return int(abs(log2(init_resolution) - log2(final_resolution)))


class MinibatchStdev(layers.Layer):
    # initialize the layer
    def __init__(self, group_size=4, **kwargs):
        super(MinibatchStdev, self).__init__(**kwargs)
        self.group_size = group_size

    # perform the operation
    def call(self, x):
        group_size = tf.minimum(self.group_size,
                                tf.shape(x)[0])  # Minibatch must be divisible by (or smaller than) group_size.
        s = x.shape  # [NCHW]  Input shape.
        y = tf.reshape(x, [group_size, -1, s[1], s[2], s[3]])  # [GMCHW] Split minibatch into M groups of size G.
        y = tf.cast(y, tf.float32)  # [GMCHW] Cast to FP32.
        y -= tf.reduce_mean(y, axis=0, keepdims=True)  # [GMCHW] Subtract mean over group.
        y = tf.reduce_mean(tf.square(y), axis=0)  # [MCHW]  Calc variance over group.
        y = tf.sqrt(y + 1e-8)  # [MCHW]  Calc stddev over group.
        y = tf.reduce_mean(y, axis=[1, 2, 3], keepdims=True)  # [M111]  Take average over fmaps and pixels.
        y = tf.cast(y, x.dtype)  # [M111]  Cast back to original data type.
        y = tf.tile(y, [group_size, 1, s[2], s[3]])  # [N1HW]  Replicate over group and pixels.
        return tf.concat([x, y], axis=1)

    # define the output shape of the layer
    def compute_output_shape(self, input_shape):
        # create a copy of the input shape as a list
        input_shape = list(input_shape)
        # add one to the channel dimension (assume channels-last)
        input_shape[-1] += 1
        # convert list to a tuple
        return tuple(input_shape)


class PixelNorm(layers.Layer):
    def __init__(self, **kwargs):
        super(PixelNorm, self).__init__(**kwargs)

    def call(self, x):
        epsilon = 1e-8
        return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


class _Generator(Model):
    KERNEL = 3

    class ConvLayer(layers.Layer):
        def __init__(self, filters, kernel, stride, w_init='glorot_uniform', w_const=None):
            super(_Generator.ConvLayer, self).__init__()
            self.filters = filters
            self.kernel = kernel
            self.stride = stride
            self.conv1 = layers.Conv2DTranspose(filters, (kernel, kernel), strides=(stride, stride), padding='same',
                                                use_bias=False, kernel_initializer=w_init, kernel_constraint=w_const)
            self.pn = PixelNorm()
            self.activation = layers.LeakyReLU()

        def call(self, X):
            Y = self.activation(self.pn(self.conv1(X)))
            return Y

    class ToRGB(Model):
        def __init__(self, num_channels, w_init='glorot_uniform', w_const=None):
            super().__init__()
            self.conv = layers.Conv2D(num_channels, (1, 1), strides=(1, 1), padding='same', kernel_initializer=w_init,
                                      kernel_constraint=w_const)

        def call(self, X, Y=None):
            t = self.conv(X)
            return t if Y is None else Y + t

    def __init__(self, latent_dim, channels, init_resolution, output_resolution, init_filters):
        super().__init__()
        print("Generator: ")
        weight_init = initializers.RandomNormal(stddev=0.02)
        weight_const = constraints.MaxNorm(max_value=1.0)
        self.num_of_conv_layers = _get_layers(init_resolution, output_resolution) - 1
        self.input_block = []
        self.conv_layers = []
        self.to_rgb = []
        self.upsample = []
        self.add = layers.Add()
        print(f"Layers: {self.num_of_conv_layers}")

        # input block
        print(f"Input: ({latent_dim}, )")
        output_shape = init_resolution * init_resolution * init_filters
        self.input_block.append(layers.Dense(output_shape, use_bias=False, input_shape=(latent_dim,),
                                             kernel_initializer=weight_init, kernel_constraint=weight_const))
        print(f"Dense output: ({init_resolution} * {init_resolution} * {init_filters})")

        # convolutional layers
        self.input_block.append(PixelNorm())
        self.input_block.append(layers.LeakyReLU())
        self.input_block.append(layers.Reshape((init_resolution, init_resolution, init_filters)))
        print(f"Reshape output: ({init_resolution}, {init_resolution}, {init_filters})")

        for i in range(self.num_of_conv_layers):
            filters = int(init_filters / 2**(i + 1))
            self.conv_layers.append(self.ConvLayer(filters, self.KERNEL, 2, w_init=weight_init, w_const=weight_const))
            self.to_rgb.append(self.ToRGB(channels, w_init=weight_init, w_const=weight_const))
            self.upsample.append(layers.UpSampling2D())
            resolution = init_resolution * 2**(i + 1)
            print(f"Conv2dTranspose output: ({resolution}, {resolution}, {filters})")

        # output layer
        self.conv_layers.append(layers.Conv2DTranspose(channels, (self.KERNEL, self.KERNEL), strides=(2, 2),
                                                       padding='same', use_bias=False, activation='tanh',
                                                       kernel_initializer=weight_init, kernel_constraint=weight_const))
        self.to_rgb.append(self.ToRGB(channels))
        resolution = init_resolution * 2**(self.num_of_conv_layers + 1)
        print(f"Conv2dTranspose output: ({resolution} * {resolution} * {channels})")

    def call(self, X):
        Y = None
        for layer in self.input_block:
            X = layer(X)

        i = 0
        layer = self.conv_layers[i]
        to_rgb = self.to_rgb[i]
        X = layer(X)
        Y = to_rgb(X, Y)
        i += 1
        while i < len(self.conv_layers):
            layer = self.conv_layers[i]
            to_rgb = self.to_rgb[i]
            up_sampling = self.upsample[i - 1]

            X = layer(X)
            Y = up_sampling(Y)
            Y = to_rgb(X, Y=Y)

            i += 1
        return Y


class Generator:
    def __init__(self, lr: float, beta_1: float, latent_dim: int, input_res, output_res, init_filers, rgb=False):
        # hyper-parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.latent_dim = latent_dim
        self.rgb = rgb
        self.input_res = input_res
        self.output_res = output_res
        self.init_filters = init_filers
        if self.rgb:
            self.channels = 3
        else:
            self.channels = 1

        # model
        self.model = None
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=self.beta_1)

    def build(self):
        self.model = _Generator(self.latent_dim, self.channels, self.input_res, self.output_res, self.init_filters)

    def loss(self, fake_score):
        return self._cross_entropy(tf.ones_like(fake_score), fake_score)


class _Discriminator(Model):
    KERNEL = 3

    class ConvLayer(layers.Layer):
        def __init__(self, filters, stride, input=None, w_init='glorot_uniform', w_const=None):
            super(_Discriminator.ConvLayer, self).__init__()
            self.filters = filters
            self.stride = stride
            if input is None:
                self.conv = layers.Conv2D(filters, (_Discriminator.KERNEL, _Discriminator.KERNEL),
                                          strides=(stride, stride), padding='same', kernel_initializer=w_init,
                                          kernel_constraint=w_const)
            else:
                self.conv = layers.Conv2D(filters, (_Discriminator.KERNEL, _Discriminator.KERNEL), strides=(2, 2),
                                          padding='same', input_shape=input, kernel_initializer=w_init,
                                          kernel_constraint=w_const)
            self.activation = layers.LeakyReLU()
            self.dropout = layers.Dropout(0.3)

        def call(self, X):
            Y = self.dropout(self.activation(self.conv(X)))
            return Y

    class FromRGB(Model):
        KERNEL = 1

        def __init__(self, filter, w_init='glorot_uniform', w_const=None):
            super().__init__()
            self.conv = layers.Conv2D(filter, kernel_size=(self.KERNEL, self.KERNEL), kernel_initializer=w_init,
                                      kernel_constraint=w_const)
            self.activation = layers.LeakyReLU()

        def call(self, X, Y=None):
            t = self.conv(Y)
            t = self.activation(t)
            return t if X is None else X + t

    class WeightedSum(layers.Add):
        # init with default value
        def __init__(self, alpha=0.0, **kwargs):
            super(_Discriminator.WeightedSum, self).__init__(**kwargs)
            self.alpha = alpha

        # output a weighted sum of inputs
        def _merge_function(self, inputs):
            # only supports a weighted sum of two inputs
            assert (len(inputs) == 2)
            # ((1-a) * input1) + (a * input2)
            output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
            return output

    def __init__(self, channels, input_resolution, final_resolution, input_filter):
        super().__init__()
        print("Discriminator: ")
        weight_init = initializers.RandomNormal(stddev=0.02)
        weight_const = constraints.MaxNorm(max_value=1.0)
        self.num_of_conv_layers = _get_layers(input_resolution, final_resolution) - 1
        self.conv_layers = []
        self.skip_layers = []
        self.output_block = []
        self.from_rgb = self.FromRGB(input_filter)

        # input block
        print(f"Input shape: ({input_resolution}, {input_resolution}, {channels})")
        self.conv_layers.append(self.ConvLayer(input_filter, 2,
                                               input=[input_resolution, input_resolution, channels],
                                               w_init=weight_init, w_const=weight_const))
        self.skip_layers.append(layers.Conv2D(input_filter, (1, 1), strides=(2, 2), padding='same',
                                              kernel_initializer=weight_init, kernel_constraint=weight_const))
        print(f"Conv2d output: {input_resolution / 2} * {input_resolution / 2}, filter: {input_filter}")

        # convolutional layers
        for i in range(self.num_of_conv_layers):
            filters = input_filter * 2**(i + 1)
            self.conv_layers.append(self.ConvLayer(filters, 2, w_init=weight_init, w_const=weight_const))
            self.skip_layers.append(layers.Conv2D(filters, (1, 1), strides=(2, 2), padding='same',
                                                  kernel_initializer=weight_init, kernel_constraint=weight_const))
            output_size = input_resolution / 2**(i + 2)
            print(f"Conv2d output: {output_size} * {output_size}, filter: {filters}")

        # output layer
        self.output_block.append(MinibatchStdev())
        self.output_block.append(layers.Conv2D(input_filter * 2**(self.num_of_conv_layers + 1),
                                               (self.KERNEL, self.KERNEL), strides=(2, 2), padding='same',
                                               kernel_initializer=weight_init, kernel_constraint=weight_const))
        self.output_block.append(layers.LeakyReLU())
        self.output_block.append(layers.Flatten())
        self.output_block.append(layers.Dense(1))
        print("Dense output: (1, )")

    def call(self, input):
        image_in, fade_in = input
        X = None
        Y = image_in
        weighted_sum = self.WeightedSum(alpha=fade_in)

        for i in range(len(self.conv_layers)):
            if i == 0:
                X = self.from_rgb(X, Y=Y)

            t = X
            X = self.conv_layers[i](X)
            t = self.skip_layers[i](t)
            X = weighted_sum([X, t])

        for layer in self.output_block:
            X = layer(X)
        return X


class Discriminator:
    def __init__(self, lr, beta_1, input_res, final_res, input_filter, rgb=False):
        # hyper-parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.input_res = input_res
        self.final_res = final_res
        self.input_filter = input_filter
        self.rgb = rgb
        if self.rgb:
            self.channels = 3
        else:
            self.channels = 1

        # model
        self.model = None
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=self.beta_1)

    def build(self):
        self.model = _Discriminator(self.channels, self.input_res, self.final_res, self.input_filter)

    def loss(self, real_score, fake_score):
        real_loss = self._cross_entropy(tf.ones_like(real_score), real_score)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_score), fake_score)
        total_loss = real_loss + fake_loss
        return total_loss
