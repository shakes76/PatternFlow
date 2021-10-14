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
    Date last modified: 13/10/2021
    Python version: 3.9.7
"""

import tensorflow as tf
import sys
import matplotlib.pyplot as plt

# Data functions
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

    # Normalise the images
    images = images.map(lambda x: x / 255.0)

    return images


def visualise_images(
    images: tf.data.Dataset, fig_size: tuple[int, int] = (16, 10)
) -> None:

    plt.figure(figsize=fig_size)
    for imgs in images.take(1):
        batch_size = imgs.shape[0]
        for i in range(min(batch_size, 32)):
            ax = plt.subplot(4, 8, i + 1)
            plt.imshow(imgs[i].numpy(), cmap="gray")
            plt.axis("off")
        plt.show()


# Error messages functions
def raise_path_error(info: str) -> None:
    to_red = "\033[91m"
    to_default = "\033[0m"
    sys.stderr.write(
        f"{to_red}{info}\nPlease make sure the file path is correct.\nIf you are on a Windows system, use '\\\\' as your file seperator.\nIf you are on a UNIX system, use '/' as your file seperator.\n{to_default}"
    )
    raise IOError(info)
