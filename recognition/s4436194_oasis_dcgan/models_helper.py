import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l1_l2
from typing import Tuple


def make_models_28() -> Tuple[Model, Model, int]:
    return Discriminator28(), Generator28(), 28


def make_models_64() -> Tuple[Model, Model, int]:
    return Discriminator64(), Generator64(), 64


def make_models_128() -> Tuple[Model, Model, int]:
    return Discriminator128(), Generator128(), 128

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
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.layer_conv_0 = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 1])
        self.layer_lrelu_0 = layers.LeakyReLU()
        self.layer_dropout_0 = layers.Dropout(0.3)

        self.layer_conv_1 = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.layer_lrelu_1 = layers.LeakyReLU()
        self.layer_dropout_1 = layers.Dropout(0.3)

        self.layer_conv_2 = layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')
        self.layer_lrelu_2 = layers.LeakyReLU()
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
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

        self.layer_input = tf.keras.Input(shape=100)
        self.layer_dense_0 = layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(100,))
        self.layer_batch_norm_0 = layers.BatchNormalization()
        self.layer_lrelu_0 = layers.LeakyReLU()

        self.layer_reshape_0 = layers.Reshape((8, 8, 256))

        self.layer_conv2d_1 = layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False)
        self.layer_batch_norm_1 = layers.BatchNormalization()
        self.layer_lrelu_1 = layers.LeakyReLU()

        self.layer_conv2d_2 = layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_2 = layers.BatchNormalization()
        self.layer_lrelu_2 = layers.LeakyReLU()

        self.layer_conv2d_3 = layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.layer_batch_norm_3 = layers.BatchNormalization()
        self.layer_lrelu_3 = layers.LeakyReLU()

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
        self.optimizer = tf.keras.optimizers.Adam(1e-4)

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
        self.layer_dense_0 = layers.Dense(8 * 8 * 512, use_bias=False, input_shape=(100,))
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
