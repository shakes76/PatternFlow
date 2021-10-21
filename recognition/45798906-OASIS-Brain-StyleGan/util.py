"""
    util.py

    Utilities for StyleGAN.

    This file contains functions to load and visualise images.

    Requirements:
        - TensorFlow 2.0
        - sys
        - Matplotlib

    Author: Keith Dao
    Date created: 13/10/2021
    Date last modified: 21/10/2021
    Python version: 3.9.7
"""

import tensorflow as tf
import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Data
def load_images(
    directories: list[str],
    batch_size: int,
    image_size: int,
) -> tf.data.Dataset:

    # Gather all the images and place into a dataset
    images = None
    for directory in directories:
        img_dataset = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            labels=None,
            image_size=[image_size, image_size],
            shuffle=True,
            batch_size=batch_size,
            color_mode="grayscale",
        )
        images = (
            img_dataset if images == None else images.concatenate(img_dataset)
        )

    if images == None:
        raise IOError("No directories were provided.")

    # Normalise the images
    images = images.map(
        lambda x: x / 255.0, num_parallel_calls=tf.data.AUTOTUNE
    )

    return images


def augment_images(images: tf.data.Dataset) -> tuple[int, tf.data.Dataset]:

    return (
        images.cardinality().numpy(),
        images.cache()
        .map(
            lambda image: tf.image.random_flip_up_down(image),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .shuffle(images.cardinality())
        .repeat(),
    )


# Visualisation
def visualise_images(
    images: tf.Tensor, fig_size: tuple[int, int] = (16, 10)
) -> None:

    batch_size = images.shape[0]
    plt.figure(figsize=fig_size)
    for i in range(min(batch_size, 32)):
        ax = plt.subplot(4, 8, i + 1)
        plt.imshow(images[i].numpy(), cmap="gray")
        plt.axis("off")
    plt.show()


def visualise_loss(
    losses: tuple[list[float], list[float]], starting_epoch: int = 0
) -> None:

    gen_losses, disc_losses = losses
    x_range = tf.range(
        starting_epoch + 1,
        starting_epoch + len(gen_losses) + 1,
    )

    # Plot
    ax = plt.gca()
    ax.plot(x_range, gen_losses, label="Generator")
    ax.plot(x_range, disc_losses, label="Discriminator")

    # Axis labels
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()

    # x axis
    plt.xlim(
        [
            0 if starting_epoch == 1 else starting_epoch,
            starting_epoch + len(gen_losses) - 1,
        ]
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.show()


# Error messages functions
def raise_path_error(info: str) -> None:
    to_red = "\033[91m"
    to_default = "\033[0m"
    sys.stderr.write(
        f"{to_red}{info}\nPlease make sure the file path is correct.\nIf you are on a Windows system, use '\\\\' as your file seperator.\nIf you are on a UNIX system, use '/' as your file seperator.\n{to_default}"
    )
    raise IOError(info)
