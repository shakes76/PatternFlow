import tensorflow as tf
import numpy as np
from matplotlib import pyplot
from tensorflow.keras.optimizers import Adam
import time
from IPython import display
from tensorflow import random, train
from numpy.random import randn, rand, randint
import os

from dataset import (
    load_images
)

from modules import (
    discriminator_model,
    generator_model,
)


def get_inputs(n, img_shape, latent_dim, n_style_block=7):
    if rand() < 0.5:
        available_z = [random.normal((n, 1, latent_dim)) for _ in range(2)]
        z = tf.concat(
            [available_z[randint(0, len(available_z))] for _ in range(n_style_block)], axis=1)
    else:
        z = tf.repeat(random.normal((n, 1, latent_dim)), n_style_block, axis=1)

    noise = random.normal(
        [n, img_shape[0], img_shape[1], 1], 0, 1, tf.float32)
    return [tf.ones((n, 1)), z, noise]


def generate_real_samples(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    # generate 'real' class labels (1)
    y = tf.ones((n_samples, 1))
    return X, y


def create_checkpoint(gen_optimizer, disc_optimizer, generator, discriminator):
    checkpoint = train.Checkpoint(generator_optimizer=gen_optimizer,
                                     discriminator_optimizer=disc_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    return checkpoint


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=9):
    # prepare fake examples
    generator_inputs = get_inputs(n_samples, dataset[0].shape, latent_dim)
    x_fake = g_model(generator_inputs, training=False)

    # save plot
    for i in range(3 * 3):
        # define subplot
        pyplot.subplot(3, 3, 1 + i)
        # turn off axis
        pyplot.axis('off')
        # plot raw pixel data
        pyplot.imshow(x_fake[i], cmap="gray")
    # save plot to file
    filename = './images/generated_plot_e%03d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()
