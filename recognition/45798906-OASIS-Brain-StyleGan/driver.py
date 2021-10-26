"""
    driver.py

    This file runs the GAN.

    Requirements:
        - Tensorflow 2.0
        - tqdm
        - matplotlib
        - util.py
        - train.py

    Author: Keith Dao
    Date created: 14/10/2021
    Date last modified: 26/10/2021
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

from util import (
    load_images,
    augment_images,
    save_figure,
    visualise_images,
    visualise_loss,
    generate_loss_history,
)
from gan import (
    get_generator,
    get_discriminator,
    get_optimizer,
    generate_samples,
    train,
)

# Training image paths
IMAGE_PATHS: list[str] = [
    # List of all image directories
    # NOTE: All paths must end with a file seperator
    "./keras_png_slices_data/unsegmented/"
]

# Training variables
TRAIN: bool = True
EPOCHS: int = 200
TOTAL_PREVIOUS_EPOCHS: int = 0  # This is set to 0 if LOAD_WEIGHTS is FALSE
MODEL_NAME: str = "Trial 12"

# Model weight loading
LOAD_WEIGHTS: bool = False
GENERATOR_WEIGHT_PATH: str = ""
DISCRIMINATOR_WEIGHT_PATH: str = ""

# Model weight saving
SAVE_WEIGHTS: bool = True
WEIGHT_SAVING_INTERVAL: int = 5
WEIGHT_PATH: str = "./weights/"

# Sample images
SAVE_SAMPLE_IMAGES: bool = True
SHOW_FINAL_SAMPLE_IMAGES: bool = False
IMAGE_SAVING_INTERVAL: int = 1
SAMPLE_IMAGES_PATH: str = "./training/"

# Model losses
VISUALISE_LOSS: bool = False
SAVE_LOSS: bool = True
LOSS_PATH: str = "./resources/"

# ==========================================================
def main():

    # Optimizers
    gen_optimizer = get_optimizer(learning_rate=1e-7)
    disc_optimizer = get_optimizer(learning_rate=1e-7)

    # Model hyperparameters
    IMAGE_SIZE: int = 256
    BATCH_SIZE: int = 32
    SAMPLE_SIZE: int = 32
    NUM_FILTERS: int = 512
    LATENT_DIMENSION: int = 512
    KERNEL_SIZE: int = 3

    # Models
    generator = get_generator(
        LATENT_DIMENSION, IMAGE_SIZE, NUM_FILTERS, KERNEL_SIZE
    )
    discriminator = get_discriminator(IMAGE_SIZE, NUM_FILTERS, KERNEL_SIZE)
    if LOAD_WEIGHTS:
        generator.load_weights(GENERATOR_WEIGHT_PATH).expect_partial()
        discriminator.load_weights(DISCRIMINATOR_WEIGHT_PATH).expect_partial()

    # Train
    if TRAIN:
        print("Loading images.")
        images = load_images(IMAGE_PATHS, BATCH_SIZE, IMAGE_SIZE)
        print("Done loading images.")

        print("Augmenting images.")
        batches, images = augment_images(images)
        print("Done augmenting images.")

        print("Starting GAN training.")
        history = train(
            generator,
            discriminator,
            gen_optimizer,
            disc_optimizer,
            images,
            LATENT_DIMENSION,
            BATCH_SIZE,
            batches,
            IMAGE_SIZE,
            EPOCHS,
            model_name=MODEL_NAME,
            epoch_offset=TOTAL_PREVIOUS_EPOCHS if LOAD_WEIGHTS else 0,
            save_weights=SAVE_WEIGHTS,
            weight_save_path=WEIGHT_PATH,
            weight_save_interval=WEIGHT_SAVING_INTERVAL,
            save_images=SAVE_SAMPLE_IMAGES,
            image_save_path=SAMPLE_IMAGES_PATH,
            image_save_interval=IMAGE_SAVING_INTERVAL,
        )
        print("Done training.")

        if VISUALISE_LOSS:
            visualise_loss(history, TOTAL_PREVIOUS_EPOCHS)

        if SAVE_LOSS:
            figure = generate_loss_history(history, TOTAL_PREVIOUS_EPOCHS)
            save_figure(
                figure,
                f"{LOSS_PATH}Loss_{'_'.join([s.lower() for s in MODEL_NAME.split()])}.png",
            )

    if SHOW_FINAL_SAMPLE_IMAGES:
        # Show sample images
        samples = generate_samples(
            generator, LATENT_DIMENSION, SAMPLE_SIZE, IMAGE_SIZE
        )
        visualise_images(samples)


if __name__ == "__main__":
    main()
