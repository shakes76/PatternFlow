"""
    driver.py

    This file runs the GAN.

    Requirements:
        - Tensorflow 2.0
        - tqdm
        - matplotlib
        - os
        - util.py
        - train.py
        - gan.py

    Author: Keith Dao
    Date created: 14/10/2021
    Date last modified: 16/10/2021
    Python version: 3.9.7
"""

import os

# Adjust TensorFlow log levels
# Obtained from: https://stackoverflow.com/a/42121886
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ[
    "TF_CPP_MIN_LOG_LEVEL"
] = "1"  # Must be done before TensorFlow import

from util import load_images, augment_images, visualise_images, visualise_loss
from gan import (
    get_generator,
    get_discriminator,
    get_optimizer,
    generate_samples,
)
from train import train


# Optimiser hyperparameters
GENERATOR_LR = 3e-5
DISCRIMINATOR_LR = 4e-5
# Additional hyperparameters
# key : hyperparameter name, value : hyperparameter value
GENERATOR_HYPERPARAMETERS = {}
DISCRIMINATOR_HYPERPARAMETERS = {}

# Model hyperparameters
IMAGE_SIZE = 256
BATCH_SIZE = 32
SAMPLE_SIZE = 32
NUM_FILTERS = 512
LATENT_DIMENSION = 512

# Paths
# End all paths with a file seperator
IMAGE_PATHS = [
    "./keras_png_slices_data/unsegmented/"
]  # List of all image directories
# The following can be set to None
SAMPLE_IMAGES_PATH = "./training/"  # If None, set SAVE_SAMPLE_IMAGES to False
WEIGHT_PATH = "./weights/"  # If None, set SAVE_WEIGHTS to False
GENERATOR_WEIGHT_PATH = ""
DISCRIMINATOR_WEIGHT_PATH = ""

# Training variables
TRAIN = True
EPOCHS = 100
TOTAL_PREVIOUS_EPOCHS = 0  # This is set to 0 if LOAD_WEIGHTS is FALSE
LOAD_WEIGHTS = False
SAVE_WEIGHTS = True
WEIGHT_SAVING_INTERVAL = 5
SAVE_SAMPLE_IMAGES = True
IMAGE_SAVING_INTERVAL = 1


def main():

    # Optimizers
    gen_optimizer = get_optimizer(
        learning_rate=GENERATOR_LR, **GENERATOR_HYPERPARAMETERS
    )
    disc_optimizer = get_optimizer(
        learning_rate=DISCRIMINATOR_LR, **DISCRIMINATOR_HYPERPARAMETERS
    )

    # Models
    generator = get_generator(
        LATENT_DIMENSION, IMAGE_SIZE, NUM_FILTERS, gen_optimizer
    )
    discriminator = get_discriminator(IMAGE_SIZE, NUM_FILTERS, disc_optimizer)
    if LOAD_WEIGHTS:
        generator.load_weights(GENERATOR_WEIGHT_PATH).expect_partial()
        discriminator.load_weights(DISCRIMINATOR_WEIGHT_PATH).expect_partial()

    # Train
    if TRAIN:
        images = load_images(IMAGE_PATHS, BATCH_SIZE, IMAGE_SIZE)
        images = augment_images(images)
        history = train(
            generator,
            discriminator,
            gen_optimizer,
            disc_optimizer,
            images,
            LATENT_DIMENSION,
            BATCH_SIZE,
            IMAGE_SIZE,
            EPOCHS,
            epoch_offset=TOTAL_PREVIOUS_EPOCHS if LOAD_WEIGHTS else 0,
            save_weights=SAVE_WEIGHTS,
            weight_save_path=WEIGHT_PATH,
            weight_save_interval=WEIGHT_SAVING_INTERVAL,
            save_images=SAVE_SAMPLE_IMAGES,
            image_save_path=SAMPLE_IMAGES_PATH,
            image_save_interval=IMAGE_SAVING_INTERVAL,
        )
        visualise_loss(history, TOTAL_PREVIOUS_EPOCHS)

    # Show sample images
    samples = generate_samples(
        generator, LATENT_DIMENSION, SAMPLE_SIZE, IMAGE_SIZE
    )
    visualise_images(samples)


if __name__ == "__main__":
    main()
