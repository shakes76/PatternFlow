import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, Conv2DTranspose, LeakyReLU, Dropout, AveragePooling2D
from tensorflow.keras.models import Sequential


def discriminator_model(kernel_size=3, input_shape=(256, 256, 3)):
    """ Create discriminator model with Inputshape = [img_size, img_size, channel] """
    initial_filter_size = initial_filter_size
    discriminator_model = Sequential([
        # Downsample the input to 128 * 128
        Conv2D(initial_filter_size, kernel_size=kernel_size,
               padding='same', input_shape=input_shape),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # Downsample the input to 64 * 64
        Conv2D(2*initial_filter_size, kernel_size=kernel_size, padding='same'),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # Downsample the input to 32 * 32
        Conv2D(4*initial_filter_size, kernel_size=kernel_size, padding='same'),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # Down sample the input to 16 * 16
        Conv2D(4*initial_filter_size, kernel_size=kernel_size, padding='same'),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # Down sample the input to 8 * 8
        Conv2D(8*initial_filter_size, kernel_size=kernel_size, padding='same'),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # Down sample the input to 4 * 4
        Conv2D(8*initial_filter_size, kernel_size=kernel_size, padding='same'),
        AveragePooling2D(),
        LeakyReLU(alpha=0.2),

        # 4 * 4 block
        Conv2D(8*initial_filter_size, kernel_size=kernel_size, padding='same'),
        LeakyReLU(alpha=0.2),

        # Flattent the input to a 1d array for classication 4*4*512 = 8192
        Flatten(),
        Dense(4*initial_filter_size),
        Dropout(0.4),
        # Dense layer with 1 output for deciding whether the input is real or fake.
        # Sigmoid -> output is between 0 and 1, closer to 1 indicates discriminator detects as real, otherwise fake
        Dense(1, activation='sigmoid'),
    ])

    return discriminator_model


def Generator_model(latent_dim=100, image_shape=(256, 256, 3)):
    num_blocks = 7  # upsample from  4*4  to 256*256
    initial_size = 4
    latent_dim = 512  # according to the paper
    num_filters = 512

    # create z layer -> can normalise the z when we generate it using randnormmal or add a normalisation layer?
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

    # create constant / starting layer : 4*4*512
    ones = Input((1,))
    const = Sequential([
        Dense(initial_size * initial_size * num_filters),
        Reshape((initial_size, initial_size, num_filters)),
    ])(ones)

    # Create noise B
    noise = Input(image_shape)
