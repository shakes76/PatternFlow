import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot
from tensorflow.keras.optimizers import Adam

from modules import (
    discriminator_model,
    generator_model,
)
LATENT_DIM = 256
CHECKPOINT_DIRECTORY = "./model_checkpoints"
IMG_SHAPE = (256, 256)
NUM_SAMPLES = 9


def get_inputs(n, img_shape, latent_dim, n_style_block=7, ):
    if np.random.rand() < 0.5:
        available_z = [tf.random.normal((n, 1, latent_dim)) for _ in range(2)]
        z = tf.concat(
            [available_z[np.random.randint(0, len(available_z))] for _ in range(n_style_block)], axis=1)
    else:
        z = tf.repeat(tf.random.normal((n, 1, latent_dim)),
                      n_style_block, axis=1)

    noise = tf.random.normal(
        [n, img_shape[0], img_shape[1], 1], 0, 1, tf.float32)
    return [tf.ones((n, 1)), z, noise]


def main():
    g_model = generator_model(latent_dim=LATENT_DIM)
    d_model = discriminator_model()
    gen_optimizer = Adam(learning_rate=2e-7, beta_1=0.5, beta_2=0.99)
    disc_optimizer = Adam(learning_rate=1.5e-7, beta_1=0.5, beta_2=0.99)

    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                     discriminator_optimizer=disc_optimizer,
                                     generator=g_model,
                                     discriminator=d_model)
    status = checkpoint.restore(
        tf.train.latest_checkpoint(CHECKPOINT_DIRECTORY))

    generator_inputs = get_inputs(NUM_SAMPLES, IMG_SHAPE, LATENT_DIM)
    x_fake = g_model(generator_inputs, training=False)
    for i in range(3 * 3):
        pyplot.subplot(3, 3, 1 + i)
        pyplot.imshow(x_fake[i], cmap="gray")
    pyplot.show()


if __name__ == '__main__':
    main()
