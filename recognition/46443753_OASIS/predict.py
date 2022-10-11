import tensorflow as tf
import os
from matplotlib import pyplot
from tensorflow.keras.optimizers import Adam

from modules import (
    discriminator_model,
    generator_model,
)

from train import (
    get_inputs
)

# Define constants
LATENT_DIM = 256
CHECKPOINT_DIRECTORY = "./model_checkpoints"
IMG_SHAPE = (256, 256)
NUM_SAMPLES = 9
PLOT_SHAPE = (3, 3)


def load_checkpoint(latent_dim, checkpoint_directory):
    """
    Load model checkpoint from given directory. 
    """
    gen_model = generator_model(latent_dim=latent_dim)
    disc_model = discriminator_model()
    gen_optimizer = Adam(learning_rate=2e-7, beta_1=0.5, beta_2=0.99)
    disc_optimizer = Adam(learning_rate=1.5e-7, beta_1=0.5, beta_2=0.99)

    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                     discriminator_optimizer=disc_optimizer,
                                     generator=gen_model,
                                     discriminator=disc_model)
    status = checkpoint.restore(
        tf.train.latest_checkpoint(checkpoint_directory))

    return gen_model, disc_model, gen_optimizer, disc_optimizer


def main():
    # Load model checkpoint
    gen_model, _, _, _ = load_checkpoint(LATENT_DIM, CHECKPOINT_DIRECTORY)
    # Generate fake inputs
    generator_inputs = get_inputs(NUM_SAMPLES, IMG_SHAPE, LATENT_DIM)

    # Generate fake images from the trained model
    x_fake = gen_model(generator_inputs, training=False)
    for i in range(PLOT_SHAPE[0] * PLOT_SHAPE[1]):
        pyplot.subplot(PLOT_SHAPE[0], PLOT_SHAPE[1], 1 + i)
        pyplot.axis('off')
        pyplot.imshow(x_fake[i], cmap="gray")
    pyplot.show()


if __name__ == '__main__':
    main()
