import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, \
    LeakyReLU, Dropout, AveragePooling2D, UpSampling2D, add, Lambda, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

def generator_optimizer(learning_rate=2e-7, beta_1=0.5, beta_2=0.99):
    gen_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    return gen_optimizer

def discriminator_optimizer(learning_rate=1.5e-7, beta_1=0.5, beta_2=0.99):
    disc_optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2)
    return disc_optimizer

def discriminator_model(filter_size=64, kernel_size=3, input_shape=(256, 256, 1)):
    """ Inputshape = [img_size, img_size, channel] """
    discriminator_model = Sequential([
        # Downsample the input to 128 * 128
        Conv2D(filter_size, kernel_size=kernel_size,
               padding='same', input_shape=input_shape),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # Downsample the input to 64 * 64
        Conv2D(2*filter_size, kernel_size=kernel_size, padding='same'),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # Downsample the input to 32 * 32
        Conv2D(2*filter_size, kernel_size=kernel_size, padding='same'),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # Down sample the input to 16 * 16
        Conv2D(4*filter_size, kernel_size=kernel_size, padding='same'),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # Down sample the input to 8 * 8
        Conv2D(4*filter_size, kernel_size=kernel_size, padding='same'),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # Down sample the input to 4 * 4
        Conv2D(8*filter_size, kernel_size=kernel_size, padding='same'),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # 4 * 4 block
        Conv2D(8*filter_size, kernel_size=kernel_size, padding='same'),
        LeakyReLU(alpha=0.2),

        # Flattent the input to a 1d array for classication 4*4*512 = 8192
        Flatten(),
        Dense(4*filter_size),
        Dropout(0.4),
        # Dense layer with 1 output for deciding whether the input is real or fake.
        # Sigmoid -> output is between 0 and 1, closer to 1 indicates discriminator detects as real, otherwise fake
        Dense(1, activation='sigmoid'),
    ])
    return discriminator_model


def AdaIN(input, epsilon=1e-8):
    x, scale, bias = input
    mean = K.mean(x, axis=[1, 2], keepdims=True)
    std = K.std(x, axis=[1, 2], keepdims=True) + epsilon
    norm = (x - mean) / std

    bias = tf.reshape(bias, (-1, 1, 1, norm.shape[-1]))
    scale = tf.reshape(scale, (-1, 1, 1, norm.shape[-1])) + 1.0
    return scale * norm + bias


def generator_block(x, noise, scale, bias, num_filters):
    noise = Dense(num_filters)(noise)
    x = Conv2D(filters=num_filters, kernel_size=3,
               strides=1, padding='same')(x)
    x = add([x, noise])
    x = Lambda(AdaIN)([x, scale, bias])
    return x


def synthesis_network(w_mapping_network, noise_, num_blocks, const, z, num_filters):
    # Synthesis network
    # Initialise synthesis network
    w = w_mapping_network(z[:, 0])

    # Affine transformation A
    scale = Dense(num_filters)(w)
    bias = Dense(num_filters)(w)

    # Noise B
    noise = Dense(num_filters)(noise_[:, :const.shape[1], :const.shape[2], :])
    x = Activation("linear")(const)

    x = add([x, noise])
    x = Lambda(AdaIN)([x, scale, bias])

    noise = Dense(num_filters)(noise_[:, :x.shape[1], :x.shape[2], :])
    x = generator_block(x, noise, scale, bias, num_filters)

    # Create rest of the blocks untill reaching the target resolution
    # add block -> repeat above and upsample each block
    for i in range(1, num_blocks):
        x = UpSampling2D()(x)
        w = w_mapping_network(z[:, i])

        scale = Dense(num_filters)(w)
        bias = Dense(num_filters)(w)

        noise = Dense(num_filters)(noise_[:, :x.shape[1], :x.shape[2], :])
        x = generator_block(x, noise, scale, bias, num_filters)

        noise = Dense(num_filters)(noise_[:, :x.shape[1], :x.shape[2], :])
        x = generator_block(x, noise, scale, bias, num_filters)
        x = LeakyReLU(0.2)(x)
    return x


def generator_model(latent_dim=100, num_filters=128, image_shape=(256, 256, 1)):
    # 7 Blcoks to upsample from  4*4  to 256*256
    num_blocks = 7
    initial_size = 4

    # create z layer
    z = Input(shape=[num_blocks, latent_dim])

    # create w Mapping network
    w_mapping_network = Sequential([
        Input(shape=[latent_dim]),
        Dense(num_filters, activation=LeakyReLU(0.2)),
        Dense(num_filters, activation=LeakyReLU(0.2)),
        Dense(num_filters, activation=LeakyReLU(0.2)),
        Dense(num_filters, activation=LeakyReLU(0.2)),
        Dense(num_filters, activation=LeakyReLU(0.2)),
        Dense(num_filters, activation=LeakyReLU(0.2)),
        Dense(num_filters, activation=LeakyReLU(0.2)),
        Dense(num_filters, activation=LeakyReLU(0.2)),
    ])

    # create constant / starting layer : 4*4*num_filters
    ones = Input((1,))
    const = Sequential([
        Dense(initial_size * initial_size * num_filters),
        Reshape((initial_size, initial_size, num_filters)),
    ])(ones)

    # Create noise B
    noise = Input(image_shape)

    # Synthesis network
    x = synthesis_network(w_mapping_network, noise,
                          num_blocks, const, z, num_filters)

    # conv2d last layer with 1 channel for grayscale. Sigmoid to output between 0, 1
    x = Conv2D(1, kernel_size=3, strides=1,
               padding="same", activation="sigmoid")(x)

    # create generator
    generator = Model(inputs=[ones, z, noise], outputs=x)
    return generator
