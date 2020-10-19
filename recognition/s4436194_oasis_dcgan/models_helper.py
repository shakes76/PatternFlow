import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.regularizers import l1_l2
from typing import Tuple


def make_models_28() -> Tuple[Model, Model, int]:
    return Discriminator28(), Generator28(), 28


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

        self.layer_dense_0 = layers.Dense(7*7*256, use_bias=False, input_shape=(100,))
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


def make_generator_model_255_255(input_shape):
    """

    Args:
        input_shape:

    Returns:

    """

    model = tf.keras.Sequential(name="keras_sequential_generator")
    model.add(layers.Dense(8*8*512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 8 * 8 * 512)

    model.add(layers.Reshape((8, 8, 512)))
    assert model.output_shape == (None, 8, 8, 512)

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 16, 16, 256)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 32, 32, 128)

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 64, 64, 64)

    model.add(layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 128, 128, 32)

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 256, 256, 1)

    return model


def make_discriminator_model_255_255(image_width, image_height):
    """

    Args:
        image_width:
        image_height:

    Returns:

    """
    reg = lambda: l1_l2(l1=1e-7, l2=1e-7)
    nch = 256
    h = 5

    model = tf.keras.Sequential(name="keras_sequential_discriminator")

    model.add(layers.Convolution2D(int(nch / 4), (h, h), padding='same', activity_regularizer=reg(),
                                   input_shape=[image_width, image_height, 1]))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    assert model.output_shape == (None, 128, 128, 64)

    model.add(layers.Convolution2D(int(nch / 2), (h, h), padding='same', activity_regularizer=reg()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    assert model.output_shape == (None, 64, 64, 128)

    model.add(layers.Convolution2D(nch, (h, h), padding='same', activity_regularizer=reg()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    assert model.output_shape == (None, 32, 32, 256)

    model.add(layers.Convolution2D(1, (3, 3), padding='valid', activity_regularizer=reg()))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    assert model.output_shape == (None, 15, 15, 1)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.Activation("sigmoid"))
    assert model.output_shape == (None, 1)

    return model


def make_generator_model_64_64(input_shape):
    """

    Args:
        input_shape:

    Returns:

    """

    model = tf.keras.Sequential(name="keras_sequential_generator")
    model.add(layers.Dense(4*4*10, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # assert model.output_shape == (None, 8 * 8 * 512)
    print(model.output_shape)

    model.add(layers.Reshape((4, 4, 10)))
    # assert model.output_shape == (None, 8, 8, 512)
    print(model.output_shape)

    model.add(layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # assert model.output_shape == (None, 16, 16, 256)
    print(model.output_shape)

    model.add(layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # assert model.output_shape == (None, 32, 32, 128)
    print(model.output_shape)

    model.add(layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # assert model.output_shape == (None, 64, 64, 64)
    print(model.output_shape)

    model.add(layers.Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 64, 64, 1)
    print(model.output_shape)

    return model


def make_discriminator_model_64_64(image_width, image_height):
    """

    Args:
        image_width:
        image_height:

    Returns:

    """
    reg = lambda: l1_l2(l1=1e-7, l2=1e-7)
    h = 5

    model = tf.keras.Sequential(name="keras_sequential_discriminator")

    model.add(layers.Convolution2D(16, (h, h), padding='same', activity_regularizer=reg(),
                                   input_shape=[image_width, image_height, 1]))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    # assert model.output_shape == (None, 128, 128, 64)
    print(model.output_shape)

    model.add(layers.Convolution2D(32, (h, h), padding='same', activity_regularizer=reg()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    # assert model.output_shape == (None, 64, 64, 128)
    print(model.output_shape)

    model.add(layers.Convolution2D(64, (h, h), padding='same', activity_regularizer=reg()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.LeakyReLU(0.2))
    # assert model.output_shape == (None, 32, 32, 256)
    print(model.output_shape)

    model.add(layers.Convolution2D(10, (2, 2), padding='same', activity_regularizer=reg()))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    # assert model.output_shape == (None, 15, 15, 1)
    print(model.output_shape)

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.Activation("sigmoid"))
    # assert model.output_shape == (None, 1)

    print(model.output_shape)

    return model

