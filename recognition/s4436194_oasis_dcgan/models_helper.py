import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1_l2


def make_generator_model():
    """
    https://github.com/bstriner/keras-adversarial/blob/master/examples/example_gan_cifar10.py
    """

    reg = lambda: l1_l2(l1=1e-7, l2=1e-7)
    nch = 256
    h = 5

    model = tf.keras.Sequential(name="keras_sequential_generator")
    model.add(layers.Dense(nch * 4 * 4, use_bias=False, input_shape=(100,), activity_regularizer=reg()))
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
    model.add(layers.Convolution2D(3, (h, h), padding="same", activity_regularizer=reg()))
    model.add(layers.Activation("sigmoid"))

    return model


def make_discriminator_model(image_width, image_height):
    reg = lambda: l1_l2(l1=1e-7, l2=1e-7)
    nch = 256
    h = 5

    model = tf.keras.Sequential(name="keras_sequential_discriminator")

    model.add(layers.Convolution2D(int(nch / 4), (h, h), padding='same', activity_regularizer=reg(),
                                   input_shape=[image_width, image_height, 3]))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.LeakyReLU(0.2))

    print(model.output_shape)

    model.add(layers.Convolution2D(int(nch / 2), (h, h), padding='same', activity_regularizer=reg()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.LeakyReLU(0.2))

    print(model.output_shape)

    model.add(layers.Convolution2D(nch, (h, h), padding='same', activity_regularizer=reg()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.LeakyReLU(0.2))

    print(model.output_shape)

    model.add(layers.Convolution2D(1, (3, 3), padding='valid', activity_regularizer=reg()))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))

    print(model.output_shape)

    model.add(layers.Flatten())
    model.add(layers.Activation("sigmoid"))

    return model
