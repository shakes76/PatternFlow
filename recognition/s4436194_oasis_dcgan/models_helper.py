"""
OASIS DCGAN Models

Main collection of models that are used for different sized DCGAN's. There are a series of classes, each representing
a generator or discriminator for a specific image size. These are bound together in function calls that create those
models for use. When using this module, only import the make_models functions.

@author nthompson97

Original GAN paper: https://arxiv.org/pdf/1511.06434.pdf
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Tuple


def make_models_28() -> Tuple[Model, Model, int]:
    return Discriminator28(), Generator28(), 28


def make_models_64() -> Tuple[Model, Model, int]:
    return Discriminator64(), Generator64(), 64


def make_models_128() -> Tuple[Model, Model, int]:
    return Discriminator128(), Generator128(), 128


def make_models_256() -> Tuple[Model, Model, int]:
    return Discriminator256(), Generator256(), 256


##########################################################################################################
# 28 x 28 models
##########################################################################################################


class Discriminator28(Model):

    def __init__(self):
        super(Discriminator28, self).__init__()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.layer_conv_0 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1])
        self.layer_lrelu_0 = layers.LeakyReLU()
        self.layer_dropout_0 = layers.Dropout(0.3)

        self.layer_conv_1 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.layer_lrelu_1 = layers.LeakyReLU()
        self.layer_dropout_1 = layers.Dropout(0.3)

        self.layer_flatten = layers.Flatten()
        self.layer_output = layers.Dense(1)

    def call(self, x):
        x = self.layer_conv_0(x)
        x = self.layer_lrelu_0(x)
        x = self.layer_dropout_0(x)

        x = self.layer_conv_1(x)
        x = self.layer_lrelu_1(x)
        x = self.layer_dropout_1(x)

        x = self.layer_flatten(x)
        return self.layer_output(x)


class Generator28(Model):

    def __init__(self):
        super(Generator28, self).__init__()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.layer_input = tf.keras.Input(shape=100)
        self.layer_dense_0 = layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,))
        self.layer_batch_norm_0 = layers.BatchNormalization()
        self.layer_lrelu_0 = layers.LeakyReLU()

        self.layer_reshape_0 = layers.Reshape((7, 7, 256))
        self.layer_conv2d_0 = layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.layer_batch_norm_1 = layers.BatchNormalization()
        self.layer_lrelu_1 = layers.LeakyReLU()

        self.layer_conv2d_1 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_2 = layers.BatchNormalization()
        self.layer_lrelu_2 = layers.LeakyReLU()

        self.layer_conv2d_2 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                                     use_bias=False, activation='tanh')

    def call(self, x):
        x = self.layer_dense_0(x)
        x = self.layer_batch_norm_0(x)
        x = self.layer_lrelu_0(x)

        x = self.layer_reshape_0(x)
        x = self.layer_conv2d_0(x)
        x = self.layer_batch_norm_1(x)
        x = self.layer_lrelu_1(x)

        x = self.layer_conv2d_1(x)
        x = self.layer_batch_norm_2(x)
        x = self.layer_lrelu_2(x)

        return self.layer_conv2d_2(x)


##########################################################################################################
# 64 x 64 models
##########################################################################################################


class Discriminator64(Model):

    def __init__(self):
        super(Discriminator64, self).__init__()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

        self.layer_conv_0 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 1])
        self.layer_lrelu_0 = layers.LeakyReLU(0.2)
        self.layer_dropout_0 = layers.Dropout(0.3)

        self.layer_conv_1 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.layer_lrelu_1 = layers.LeakyReLU(0.2)
        self.layer_dropout_1 = layers.Dropout(0.3)

        self.layer_conv_2 = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')
        self.layer_lrelu_2 = layers.LeakyReLU(0.2)
        self.layer_dropout_2 = layers.Dropout(0.3)

        self.layer_flatten = layers.Flatten()
        self.layer_output = layers.Dense(1)

    def call(self, x):
        x = self.layer_conv_0(x)
        x = self.layer_lrelu_0(x)
        x = self.layer_dropout_0(x)

        x = self.layer_conv_1(x)
        x = self.layer_lrelu_1(x)
        x = self.layer_dropout_1(x)

        x = self.layer_conv_2(x)
        x = self.layer_lrelu_2(x)
        x = self.layer_dropout_2(x)

        x = self.layer_flatten(x)
        return self.layer_output(x)


class Generator64(Model):

    def __init__(self):
        super(Generator64, self).__init__()

        # Test the model architecture
        # self.network_check()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

        self.layer_input = tf.keras.Input(shape=100)
        self.layer_dense_0 = layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,))
        self.layer_batch_norm_0 = layers.BatchNormalization()
        self.layer_lrelu_0 = layers.ReLU()

        self.layer_reshape_0 = layers.Reshape((8, 8, 256))

        self.layer_conv2d_1 = layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.layer_batch_norm_1 = layers.BatchNormalization()
        self.layer_lrelu_1 = layers.ReLU()

        self.layer_conv2d_2 = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_2 = layers.BatchNormalization()
        self.layer_lrelu_2 = layers.ReLU()

        self.layer_conv2d_3 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_3 = layers.BatchNormalization()
        self.layer_lrelu_3 = layers.ReLU()

        self.layer_conv2d_4 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                                     use_bias=False, activation='tanh')

    def call(self, x):
        x = self.layer_dense_0(x)
        x = self.layer_batch_norm_0(x)
        x = self.layer_lrelu_0(x)

        x = self.layer_reshape_0(x)

        x = self.layer_conv2d_1(x)
        x = self.layer_batch_norm_1(x)
        x = self.layer_lrelu_1(x)

        x = self.layer_conv2d_2(x)
        x = self.layer_batch_norm_2(x)
        x = self.layer_lrelu_2(x)

        x = self.layer_conv2d_3(x)
        x = self.layer_batch_norm_3(x)
        x = self.layer_lrelu_3(x)

        return self.layer_conv2d_4(x)

    @staticmethod
    def network_check():
        """Create a replica model and test structure"""

        model = tf.keras.Sequential(name="keras_sequential_generator")
        model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Reshape((8, 8, 256)))
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        print(model.output_shape)

        return


##########################################################################################################
# 128 x 128 models
##########################################################################################################


class Discriminator128(Model):

    def __init__(self):
        super(Discriminator128, self).__init__()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.SGD()

        self.layer_conv_0 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 1])
        self.layer_lrelu_0 = layers.LeakyReLU()
        self.layer_dropout_0 = layers.Dropout(0.3)

        self.layer_conv_1 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.layer_lrelu_1 = layers.LeakyReLU()
        self.layer_dropout_1 = layers.Dropout(0.3)

        self.layer_conv_2 = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')
        self.layer_lrelu_2 = layers.LeakyReLU()
        self.layer_dropout_2 = layers.Dropout(0.3)

        self.layer_conv_3 = layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same')
        self.layer_lrelu_3 = layers.LeakyReLU()
        self.layer_dropout_3 = layers.Dropout(0.3)

        self.layer_flatten = layers.Flatten()
        self.layer_output = layers.Dense(1)

    def call(self, x):
        x = self.layer_conv_0(x)
        x = self.layer_lrelu_0(x)
        x = self.layer_dropout_0(x)

        x = self.layer_conv_1(x)
        x = self.layer_lrelu_1(x)
        x = self.layer_dropout_1(x)

        x = self.layer_conv_2(x)
        x = self.layer_lrelu_2(x)
        x = self.layer_dropout_2(x)

        x = self.layer_conv_3(x)
        x = self.layer_lrelu_3(x)
        x = self.layer_dropout_3(x)

        x = self.layer_flatten(x)
        return self.layer_output(x)


class Generator128(Model):

    def __init__(self):
        super(Generator128, self).__init__()

        # Test the model architecture
        # self.network_check()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.layer_input = tf.keras.Input(shape=100)
        self.layer_dense_0 = layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(256,))
        self.layer_batch_norm_0 = layers.BatchNormalization()
        self.layer_lrelu_0 = layers.LeakyReLU()

        self.layer_reshape_0 = layers.Reshape((8, 8, 512))

        self.layer_conv2d_1 = layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.layer_batch_norm_1 = layers.BatchNormalization()
        self.layer_lrelu_1 = layers.LeakyReLU()

        self.layer_conv2d_2 = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_2 = layers.BatchNormalization()
        self.layer_lrelu_2 = layers.LeakyReLU()

        self.layer_conv2d_3 = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_3 = layers.BatchNormalization()
        self.layer_lrelu_3 = layers.LeakyReLU()

        self.layer_conv2d_4 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_4 = layers.BatchNormalization()
        self.layer_lrelu_4 = layers.LeakyReLU()

        self.layer_conv2d_5 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                                     use_bias=False, activation='tanh')

    def call(self, x):
        x = self.layer_dense_0(x)
        x = self.layer_batch_norm_0(x)
        x = self.layer_lrelu_0(x)

        x = self.layer_reshape_0(x)

        x = self.layer_conv2d_1(x)
        x = self.layer_batch_norm_1(x)
        x = self.layer_lrelu_1(x)

        x = self.layer_conv2d_2(x)
        x = self.layer_batch_norm_2(x)
        x = self.layer_lrelu_2(x)

        x = self.layer_conv2d_3(x)
        x = self.layer_batch_norm_3(x)
        x = self.layer_lrelu_3(x)

        x = self.layer_conv2d_4(x)
        x = self.layer_batch_norm_4(x)
        x = self.layer_lrelu_4(x)

        return self.layer_conv2d_5(x)

    @staticmethod
    def network_check():
        """Create a replica model and test structure"""

        model = tf.keras.Sequential(name="keras_sequential_generator")
        model.add(layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Reshape((8, 8, 512)))
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(512, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        print(model.output_shape)

        return


##########################################################################################################
# 256 x 256 models
##########################################################################################################


class Discriminator256(Model):

    def __init__(self):
        super(Discriminator256, self).__init__()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        self.layer_conv_0 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[256, 256, 1])
        self.layer_batch_norm_0 = layers.BatchNormalization()
        self.layer_lrelu_0 = layers.LeakyReLU(alpha=0.2)
        self.layer_dropout_0 = layers.Dropout(0.2)

        self.layer_conv_1 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.layer_batch_norm_1 = layers.BatchNormalization()
        self.layer_lrelu_1 = layers.LeakyReLU(alpha=0.2)
        self.layer_dropout_1 = layers.Dropout(0.2)

        self.layer_conv_2 = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')
        self.layer_batch_norm_2 = layers.BatchNormalization()
        self.layer_lrelu_2 = layers.LeakyReLU(alpha=0.2)
        self.layer_dropout_2 = layers.Dropout(0.2)

        self.layer_conv_3 = layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same')
        self.layer_batch_norm_3 = layers.BatchNormalization()
        self.layer_lrelu_3 = layers.LeakyReLU(alpha=0.2)
        self.layer_dropout_3 = layers.Dropout(0.2)

        self.layer_conv_4 = layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same')
        self.layer_batch_norm_4 = layers.BatchNormalization()
        self.layer_lrelu_4 = layers.LeakyReLU(alpha=0.2)
        self.layer_dropout_4 = layers.Dropout(0.2)

        self.layer_flatten = layers.Flatten()
        self.layer_output = layers.Dense(1)

    def call(self, x):
        x = self.layer_conv_0(x)
        x = self.layer_batch_norm_0(x)
        x = self.layer_lrelu_0(x)
        x = self.layer_dropout_0(x)

        x = self.layer_conv_1(x)
        x = self.layer_batch_norm_1(x)
        x = self.layer_lrelu_1(x)
        x = self.layer_dropout_1(x)

        x = self.layer_conv_2(x)
        x = self.layer_batch_norm_2(x)
        x = self.layer_lrelu_2(x)
        x = self.layer_dropout_2(x)

        x = self.layer_conv_3(x)
        x = self.layer_batch_norm_3(x)
        x = self.layer_lrelu_3(x)
        x = self.layer_dropout_3(x)

        x = self.layer_conv_4(x)
        x = self.layer_batch_norm_4(x)
        x = self.layer_lrelu_4(x)
        x = self.layer_dropout_4(x)

        x = self.layer_flatten(x)
        return self.layer_output(x)


class Generator256(Model):

    def __init__(self):
        super(Generator256, self).__init__()

        # Test the model architecture
        self.network_check()

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        self.layer_input = tf.keras.Input(shape=100)
        self.layer_dense_0 = layers.Dense(8 * 8 * 2048, use_bias=False, input_shape=(100,))
        self.layer_batch_norm_0 = layers.BatchNormalization()
        self.layer_lrelu_0 = layers.ReLU()

        self.layer_reshape_0 = layers.Reshape((8, 8, 2048))

        self.layer_conv2d_1 = layers.Conv2DTranspose(2048, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.layer_batch_norm_1 = layers.BatchNormalization()
        self.layer_lrelu_1 = layers.ReLU()

        self.layer_conv2d_2 = layers.Conv2DTranspose(1024, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_2 = layers.BatchNormalization()
        self.layer_lrelu_2 = layers.ReLU()

        self.layer_conv2d_3 = layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_3 = layers.BatchNormalization()
        self.layer_lrelu_3 = layers.ReLU()

        self.layer_conv2d_4 = layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_4 = layers.BatchNormalization()
        self.layer_lrelu_4 = layers.ReLU()

        self.layer_conv2d_5 = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_5 = layers.BatchNormalization()
        self.layer_lrelu_5 = layers.ReLU()

        self.layer_conv2d_6 = layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same',
                                                     use_bias=False, activation='tanh')

    def call(self, x):
        x = self.layer_dense_0(x)
        x = self.layer_batch_norm_0(x)
        x = self.layer_lrelu_0(x)

        x = self.layer_reshape_0(x)

        x = self.layer_conv2d_1(x)
        x = self.layer_batch_norm_1(x)
        x = self.layer_lrelu_1(x)

        x = self.layer_conv2d_2(x)
        x = self.layer_batch_norm_2(x)
        x = self.layer_lrelu_2(x)

        x = self.layer_conv2d_3(x)
        x = self.layer_batch_norm_3(x)
        x = self.layer_lrelu_3(x)

        x = self.layer_conv2d_4(x)
        x = self.layer_batch_norm_4(x)
        x = self.layer_lrelu_4(x)

        x = self.layer_conv2d_5(x)
        x = self.layer_batch_norm_5(x)
        x = self.layer_lrelu_5(x)

        return self.layer_conv2d_6(x)

    @staticmethod
    def network_check():
        """Create a replica model and test structure"""

        model = tf.keras.Sequential(name="keras_sequential_generator")
        model.add(layers.Dense(8 * 8 * 1024, use_bias=False, input_shape=(100,)))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Reshape((8, 8, 1024)))
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(1024, (3, 3), strides=(1, 1), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU())
        print(model.output_shape)

        model.add(layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
        print(model.output_shape)

        return
