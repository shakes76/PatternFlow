# !/user/bin/env python
"""
The generator and discriminator models of the StyleGAN
"""

from math import log2
import tensorflow as tf
from tensorflow.keras import initializers, layers, Model, constraints

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"


def _get_layers(init_resolution, final_resolution):
    return int(abs(log2(init_resolution) - log2(final_resolution)))


#####################################################################
# self-defined layers used in generator and discriminator
#####################################################################
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


class ToRGB(Model):
    """
    RGB to high-dimensional per-pixel data
    """

    def __init__(self,
                 num_channels,  # number of channels in the output
                 w_init='glorot_uniform',  # kernel initializer
                 w_const=None):  # kernel constraint

        super().__init__()
        self.conv = layers.Conv2D(num_channels, (1, 1), strides=(1, 1), padding='same', kernel_initializer=w_init,
                                  kernel_constraint=w_const)

    def call(self, X, Y=None):
        t = self.conv(X)
        return t if Y is None else Y + t


class FromRGB(Model):
    """
    High-dimension per-pixel data back to RGB
    """
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
    """
    Interpolation between two images
    """

    def __init__(self, alpha=0.0, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)
        self.alpha = alpha  # the portion of left image during interpolation

    # output a weighted sum of inputs
    def _merge_function(self, inputs):
        # only supports a weighted sum of two inputs
        assert (len(inputs) == 2)
        # ((1-a) * input1) + (a * input2)
        output = ((1.0 - self.alpha) * inputs[0]) + (self.alpha * inputs[1])
        return output


#####################################################################
# Generator and discriminator
#####################################################################
class _Generator(Model):
    """
    The underlying generator model
    """
    KERNEL = 3

    class ConvLayer(layers.Layer):
        def __init__(self,
                     filters,                       # number of filters in the output of the convolution
                     kernel,                        # kernel size in the convolution, e.g. (3, 3) kernel -> kernel=3
                     stride,                        # stride size in the convolution, e.g. (1, 1) stride -> stride=1
                     w_init='glorot_uniform',       # kernel initializer
                     w_const=None):                 # kernel constraints

            super(_Generator.ConvLayer, self).__init__()
            self.filters = filters
            self.kernel = kernel
            self.stride = stride

            # convolution block
            self.conv1 = layers.Conv2DTranspose(filters, (kernel, kernel), strides=(stride, stride), padding='same',
                                                use_bias=False, kernel_initializer=w_init, kernel_constraint=w_const)
            self.pn = PixelNorm()
            self.activation = layers.LeakyReLU()

        def call(self, X):
            Y = self.activation(self.pn(self.conv1(X)))
            return Y

    def __init__(self,
                 latent_dim,            # length of the input latent
                 channels,              # number of channels in the output images
                 init_resolution,       # resolution that the input latent is reshaped to, must be a power of 2
                 output_resolution,     # resolution of the output images, must be a power of 2
                 init_filters):         # number of filters starting from which will be doubled at each convolutional
                                        #  layer

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
            self.to_rgb.append(ToRGB(channels, w_init=weight_init, w_const=weight_const))
            self.upsample.append(layers.UpSampling2D())
            resolution = init_resolution * 2**(i + 1)
            print(f"Conv2dTranspose output: ({resolution}, {resolution}, {filters})")

        # output layer
        self.conv_layers.append(layers.Conv2DTranspose(channels, (self.KERNEL, self.KERNEL), strides=(2, 2),
                                                       padding='same', use_bias=False, activation='tanh',
                                                       kernel_initializer=weight_init, kernel_constraint=weight_const))
        self.to_rgb.append(ToRGB(channels))
        resolution = init_resolution * 2**(self.num_of_conv_layers + 1)
        print(f"Conv2dTranspose output: ({resolution} * {resolution} * {channels})")

    def call(self, X):
        Y = None
        for layer in self.input_block:
            X = layer(X)

        # the first convolutional layer
        i = 0
        layer = self.conv_layers[i]
        to_rgb = self.to_rgb[i]
        X = layer(X)
        Y = to_rgb(X, Y)
        i += 1

        # iterate through the rest of the convolutional layers
        while i < len(self.conv_layers):
            layer = self.conv_layers[i]
            to_rgb = self.to_rgb[i]
            up_sampling = self.upsample[i - 1]

            X = layer(X)
            Y = up_sampling(Y)
            # blend in the high-level representation of the last resolution block
            Y = to_rgb(X, Y=Y)

            i += 1

        return Y


class Generator:
    """
    The wrapper class of the generator.
    """
    def __init__(self,
                 lr: float,         # learning rate of the optimizer
                 beta_1: float,     # exponential decay rate for the first moment estimate
                 latent_dim: int,   # length of the input latent
                 input_res: int,    # resolution at the first convolutional layer
                 output_res,        # output resolution
                 init_filers,       # number of filters starting from which will be doubled at each convolutional
                                    #  layer

                 rgb=False):        # whether the output image is in RGB or not

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
        """
        Initialize the generator
        """
        self.model = _Generator(self.latent_dim, self.channels, self.input_res, self.output_res, self.init_filters)

    def loss(self, fake_score):
        """
        The cross entropy loss of the generator

        :param fake_score: The generator output for generated images
        :return: cross entropy loss
        """
        return self._cross_entropy(tf.ones_like(fake_score), fake_score)


class _Discriminator(Model):
    """
    The underlying discriminator model
    """
    KERNEL = 3

    class ConvLayer(layers.Layer):
        def __init__(self,
                     filters: int,                   # number of output filters
                     stride: int,                    # number of strides, e.g. (1, 1) stride -> stride=1
                     input=None,                     # the input size if it is the input layer
                     w_init='glorot_uniform',        # kernel initializer
                     w_const=None):                  # kernel constraint

            super(_Discriminator.ConvLayer, self).__init__()
            self.filters = filters
            self.stride = stride

            # convolutional block
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

    def __init__(self, channels, input_resolution, final_resolution, input_filter):
        super().__init__()
        print("Discriminator: ")
        # weight initializer and constraint for the convolutional layers
        weight_init = initializers.RandomNormal(stddev=0.02)
        weight_const = constraints.MaxNorm(max_value=1.0)
        self.num_of_conv_layers = _get_layers(input_resolution, final_resolution) - 1
        self.conv_layers = []
        self.skip_layers = []
        self.output_block = []
        self.from_rgb = FromRGB(input_filter)

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
        weighted_sum = WeightedSum(alpha=fade_in)

        # go through each convolutional layer
        for i in range(len(self.conv_layers)):
            if i == 0:
                X = self.from_rgb(X, Y=Y)

            t = X
            X = self.conv_layers[i](X)

            # blend in the generated image of this layer and the last layer
            t = self.skip_layers[i](t)
            X = weighted_sum([X, t])

        for layer in self.output_block:
            X = layer(X)
        return X


class Discriminator:
    """
    The wrapper class of the discriminator
    """
    def __init__(self,
                 lr: float,             # learning rate of the optimizer
                 beta_1: float,         # exponential decay rate for the first moment estimate
                 input_res: int,        # resolution at the first convolutional layer
                 final_res: int,        # output resolution
                 input_filter: int,     # number of filters starting from which will be halved at each convolutional
                                        #  layer
                 rgb=False):            # whether the output image is in RGB or not

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
        """
        Initialize the discriminator
        """
        self.model = _Discriminator(self.channels, self.input_res, self.final_res, self.input_filter)

    def loss(self, real_score, fake_score):
        """
        The cross-entropy loss of the discriminator.

        :param real_score: the generator output for the training batch
        :param fake_score: the generator output for the generated images
        :return: cross entropy loss
        """
        real_loss = self._cross_entropy(tf.ones_like(real_score), real_score)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_score), fake_score)
        total_loss = real_loss + fake_loss
        return total_loss
