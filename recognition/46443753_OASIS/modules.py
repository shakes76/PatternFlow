import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Conv2D, \
    LeakyReLU, Dropout, AveragePooling2D, UpSampling2D, add, Lambda, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def generator_optimizer(learning_rate=2e-7, beta_1=0.5, beta_2=0.99):
    """
    Define generator's Adam optimizer with default learning rate 2e-7. 
    """
    gen_optimizer = Adam(learning_rate=learning_rate,
                         beta_1=beta_1, beta_2=beta_2)
    return gen_optimizer


def discriminator_optimizer(learning_rate=1.5e-7, beta_1=0.5, beta_2=0.99):
    """
    Define discriminator's Adam optimizer with default learning rate 1.5e-7
    """
    disc_optimizer = Adam(learning_rate=learning_rate,
                          beta_1=beta_1, beta_2=beta_2)
    return disc_optimizer


def discriminator_model(filter_size=64, kernel_size=3, input_shape=(256, 256, 1)):
    """ 
    Define discriminator model with defualt filter size of 64 and input shape
    = [img_size, img_size, channel] = (256, 256, 1)
    """
    discriminator_model = Sequential([
        # Downsample the input from 256*256 to 128 * 128
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

        # flatten the input to a 1d array for classication 4*4*512 = 8192
        Flatten(),
        Dense(4*filter_size),
        Dropout(0.4),
        # Dense layer with 1 output for deciding whether the input is real or fake.
        # Sigmoid -> output is between 0 and 1, closer to 1 indicates discriminator detects as real, otherwise fake
        Dense(1, activation='sigmoid'),
    ])
    return discriminator_model


def AdaIN(input, epsilon=1e-8):
    """
    Define custom AdaIN (adaptive instance normalisation) layer according to
    the reference paper. 
    "A Style-Based Generator Architecture for Generative Adversarial Networks"

    This is defined as:
    AdaIn(x_i, y) = scale * (norm_x) + bias
    """
    x, scale, bias = input
    # calculate norm of input
    mean = K.mean(x, axis=[1, 2], keepdims=True)
    # add epsilon to avoid division by 0
    std = K.std(x, axis=[1, 2], keepdims=True) + epsilon
    norm = (x - mean) / std

    scale = tf.reshape(scale, (-1, 1, 1, norm.shape[-1])) + 1.0
    bias = tf.reshape(bias, (-1, 1, 1, norm.shape[-1]))
    return scale * norm + bias


def generator_block(x, noise, scale, bias, num_filters):
    """
    Define single generator block according to the reference paper with a 
    convolution layer added with some noise, and feeds to an AdaIN layer. 

    """
    noise = Dense(num_filters)(noise)
    x = Conv2D(filters=num_filters, kernel_size=3,
               strides=1, padding='same')(x)
    x = add([x, noise])
    x = Lambda(AdaIN)([x, scale, bias])
    return x


def synthesis_network(w_mapping_network, noise_, num_blocks, const, z, num_filters):
    """
    Define the synthesis network given some noise and latent space w. 
    Given the number of blocks, the network will grow from the initial 4*4 size
    untill the target resolution (7 blocks -> 256*256).
    """
    # Initialise synthesis network
    w = w_mapping_network(z[:, 0])

    # Affine transformation A
    scale = Dense(num_filters)(w)
    bias = Dense(num_filters)(w)

    # Noise B
    noise = Dense(num_filters)(noise_[:, :const.shape[1], :const.shape[2], :])

    # Apply noise and adaIN layer in first block
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

        # Add two generator blocks in each upsampling of the model
        noise = Dense(num_filters)(noise_[:, :x.shape[1], :x.shape[2], :])
        x = generator_block(x, noise, scale, bias, num_filters)

        noise = Dense(num_filters)(noise_[:, :x.shape[1], :x.shape[2], :])
        x = generator_block(x, noise, scale, bias, num_filters)
        x = LeakyReLU(0.2)(x)

    return x


def generator_model(latent_dim=100, num_filters=256, image_shape=(256, 256, 1), num_blocks=7, initial_size=4):
    """
    Define StyleGan generator model with given latent dim, num_filters and image shape. 
    """
    # create z layer
    # 7 Blcoks to upsample from  4*4  to 256*256
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
