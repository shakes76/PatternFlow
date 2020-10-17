import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2


def make_generator_model(input_shape):
    """
    https://github.com/bstriner/keras-adversarial/blob/master/examples/example_gan_cifar10.py

    Args:
        input_shape:

    Returns:

    """

    reg = lambda: l1_l2(l1=1e-7, l2=1e-7)
    nch = 256
    h = 5

    model = tf.keras.Sequential(name="keras_sequential_generator")
    model.add(layers.Dense(nch * 4 * 4, use_bias=False, input_shape=(input_shape,), activity_regularizer=reg()))
    model.add(layers.BatchNormalization())
    model.add(layers.Reshape((nch, 4, 4)))

    print(model.output_shape)

    model.add(layers.Convolution2D(int(nch / 2), (h, h), padding="same", activity_regularizer=reg()))
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.LeakyReLU(0.2))

    print(model.output_shape)

    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Convolution2D(int(nch / 2), (h, h), padding="same", activity_regularizer=reg()))
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.LeakyReLU(0.2))

    print(model.output_shape)

    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Convolution2D(int(nch / 4), (h, h), padding="same", activity_regularizer=reg()))
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.LeakyReLU(0.2))

    print(model.output_shape)

    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Convolution2D(int(nch / 4), (h, h), padding="same", activity_regularizer=reg()))
    model.add(layers.BatchNormalization(axis=1))
    model.add(layers.LeakyReLU(0.2))

    print(model.output_shape)

    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Convolution2D(3, (h, h), padding="same", activity_regularizer=reg()))
    model.add(layers.Activation("sigmoid"))

    print(model.output_shape)

    return model


def make_generator_model_basic(input_shape):
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

    print(model.output_shape)

    return model


def make_discriminator_model(image_width, image_height):
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
