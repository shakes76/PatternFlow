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
    Date last modified: 29/10/2021
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
from model import (
    get_generator,
    get_discriminator,
    get_optimizer,
    generate_samples,
    train,
)

# Training variables
TRAIN: bool = True
IMAGE_PATHS: list[str] = [
    # List of all image directories
    # NOTE: All paths must end with a file separator
    "./keras_png_slices_data/unsegmented/"
]
EPOCHS: int = 300
TOTAL_PREVIOUS_EPOCHS: int = 0  # This is set to 0 if LOAD_WEIGHTS is FALSE
MODEL_NAME: str = "Final"

# Model weight variables
# Model weight loading
LOAD_WEIGHTS: bool = False
GENERATOR_WEIGHT_PATH: str = ""
DISCRIMINATOR_WEIGHT_PATH: str = ""
# Model weight saving
SAVE_WEIGHTS: bool = True
WEIGHT_SAVING_INTERVAL: int = EPOCHS
WEIGHT_PATH: str = "./weights/"

# Image sampling variables
SHOW_FINAL_SAMPLE_IMAGES: bool = False
SAVE_SAMPLE_IMAGES: bool = True
IMAGE_SAVING_INTERVAL: int = 1
SAMPLE_IMAGES_PATH: str = "./training/"

# Model loss visualisation variables
VISUALISE_LOSS: bool = False
SAVE_LOSS: bool = True
LOSS_PATH: str = "./resources/"

# ==========================================================
def main():

    # Optimizers
    gen_optimizer = disc_optimizer = None
    if TRAIN:
        gen_optimizer = get_optimizer(
            learning_rate=8e-7, beta_1=0.5, beta_2=0.999
        )
        disc_optimizer = get_optimizer(
            learning_rate=1e-7, beta_1=0.5, beta_2=0.999
        )
        print("Loaded optimizers.")

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
    print("Loaded generator model.")
    if LOAD_WEIGHTS:
        generator.load_weights(GENERATOR_WEIGHT_PATH).expect_partial()
        print("Loaded generator weights.")

    discriminator = None
    if TRAIN:
        discriminator = get_discriminator(IMAGE_SIZE, NUM_FILTERS, KERNEL_SIZE)
        print("Loaded discriminator model.")
        if LOAD_WEIGHTS:
            discriminator.load_weights(
                DISCRIMINATOR_WEIGHT_PATH
            ).expect_partial()
            print("Loaded discriminator weights.")

    # Train
    if TRAIN:
        images = load_images(IMAGE_PATHS, BATCH_SIZE, IMAGE_SIZE)
        print("Loaded training images.")

        batches, images = augment_images(images)
        print("Augmented images.")

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
            print("Preparing loss history visualisation.")
            print("Opening visualisation window.")
            visualise_loss(history, TOTAL_PREVIOUS_EPOCHS)
            print("Loss history visualisation closed.")

        if SAVE_LOSS:
            figure = generate_loss_history(history, TOTAL_PREVIOUS_EPOCHS)
            loss_save_path = f"{LOSS_PATH}Loss_{'_'.join([s.lower() for s in MODEL_NAME.split()])}_{(TOTAL_PREVIOUS_EPOCHS // EPOCHS) + 1}.png"
            save_figure(figure, loss_save_path)
            print(f"Saved loss history to {loss_save_path}.")

    if SHOW_FINAL_SAMPLE_IMAGES:
        print("Perparing samples of the model.")
        samples = generate_samples(
            generator, LATENT_DIMENSION, SAMPLE_SIZE, IMAGE_SIZE
        )
        print("Opening visualisation window.")
        visualise_images(samples)
        print("Sample visualisation closed.")
    print("Script ending.")


if __name__ == "__main__":
    main()
