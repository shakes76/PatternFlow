"""
    util.py

    Utilities for StyleGAN.

    This file contains functions to load and visualise images.

    Requirements:
        - TensorFlow 2.0
        - Matplotlib

    Author: Keith Dao
    Date created: 13/10/2021
    Date last modified: 29/10/2021
    Python version: 3.9.7
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Data
def load_images(
    directories: list[str],
    batch_size: int,
    image_size: int,
) -> tf.data.Dataset:
    """Load the images in the given directories to a Tensorflow dataset."""

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

    # Dataset validation
    if images == None:
        raise IOError("No directories were provided.")
    if images.cardinality() == 0:
        raise IOError("Provided directories did not contain any images.")

    # Normalise the images from [0, 255] to [0, 1]
    images = images.map(
        lambda x: x / 255.0, num_parallel_calls=tf.data.AUTOTUNE
    )

    return images


def augment_images(images: tf.data.Dataset) -> tuple[int, tf.data.Dataset]:
    """Add augmentation to the images in the dataset."""

    return (
        # Number of batches
        images.cardinality().numpy(),
        # Training images
        images.cache()
        .map(
            lambda image: tf.image.random_flip_up_down(image),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .shuffle(images.cardinality())
        .repeat(),
    )


# ==========================================================
# Visualisation generation
def generate_image_grid(
    images: tf.Tensor, fig_size: tuple[int, int] = (16, 10)
) -> plt.Figure:
    """Create a grid of given images in a matplotlib figure."""

    batch_size = images.shape[0]
    figure = plt.figure(figsize=fig_size)
    for i in range(min(batch_size, 32)):
        ax = plt.subplot(4, 8, i + 1)
        plt.imshow(images[i].numpy(), cmap="gray")
        plt.axis("off")
    return figure


def generate_loss_history(
    losses: tuple[list[float], list[float]], starting_epoch: int = 0
) -> plt.Figure:
    """Plot the loss history."""

    figure = plt.figure()

    # Extract and setup data
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

    return figure


# ==========================================================
# Visualisation
def visualise_images(
    images: tf.Tensor, fig_size: tuple[int, int] = (16, 10)
) -> None:
    """Generate and show the images in a grid"""

    figure = generate_image_grid(images, fig_size)
    plt.show()


def visualise_loss(
    losses: tuple[list[float], list[float]], starting_epoch: int = 0
) -> None:
    """Generate and show the loss history."""

    figure = generate_loss_history(losses, starting_epoch)
    plt.show()


# ==========================================================
# Saving
def save_figure(figure: plt.Figure, file_path: str) -> None:
    """Save the given figure to the file path."""

    figure.savefig(file_path, bbox_inches="tight")
    plt.close(figure)
