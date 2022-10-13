"""dataset.py

Contains functions for loading and preprocessing the ADNI-MRI data.
"""

import os
from typing import Tuple
from typing import Any

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from keras.utils import image_dataset_from_directory

from constants import IMG_ORIG_SIZE, IMG_DOWN_SIZE, BATCH_SIZE, RESIZE_METHOD


def download_data() -> str:
    """Download the ADNI-MRI data and return the resulting download path

    The folder path returned should have the following folders inside:
        test/
            AD/
                _images_.jpeg
            NC/
                _images_.jpeg
        train/
            AD/
                _images_.jpeg
            NC/
                _images_.jpeg

    Returns:
        (str): String form of the path that this data is downloaded to
    """
    dataset_url = \
        "https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download"
    data_dir = keras.utils.get_file(
        origin=dataset_url,
        fname="ADNI-MRI",
        extract=True
    )
    root_dir = os.path.join(data_dir, "../AD_NC")

    return root_dir


def get_datasets(data_path: str) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Return normalised train and test datasets from data_path

    data_path must have two subdirectories: test and train representing the
    testing and training data sets (shown below).
        test/
            AD/
                _images_.jpeg
            NC/
                _images_.jpeg
        train/
            AD/
                _images_.jpeg
            NC/
                _images_.jpeg

    The shapes of the datasets will be ((64, 60, 1), (256, 240, 1)), i.e.
    downsampled images mapped to full-res target images.

    Args:
        data_path (str): path to the folder containing the images

    Returns:
        List[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Tuple of
        train, valid, and test datasets (train, valid, test)
    """
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    train_ds = image_dataset_from_directory(
        train_path,
        image_size=IMG_ORIG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode=None,
    )

    test_ds = image_dataset_from_directory(
        test_path,
        image_size=IMG_ORIG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        label_mode=None,
    )

    normalisation_layer = keras.layers.Rescaling(1./255)  # Normalise

    normalised_sets = []

    # Normalise and downsample each image and insert its target
    for dataset in [train_ds, test_ds]:
        normalised_sets.append(dataset.map(lambda x: (
            tf.image.resize(normalisation_layer(x), IMG_DOWN_SIZE, 
                            method=RESIZE_METHOD),
            normalisation_layer(x)
        )))

    return tuple(normalised_sets)


def downsample_image(
    image: Any,
    resulting_size: tuple[int, int] = IMG_DOWN_SIZE,
) -> Any:
    """Return image downsized to resulting_size.

    Args:
        image (Any): image to downsize.
        resulting_size tuple[int, int]: The image size to downsample towards.
            Defaults to (64, 60).

    Returns:
        Any: Downsized image
    """
    return tf.image.resize(image, resulting_size, method=RESIZE_METHOD)


def preview_data(dataset: tf.data.Dataset, title: str | None = None) -> None:
    """Construct a matplotlib figure to preview some given image dataset

    Args:
        dataset (tf.data.Dataset): Dataset to preview
        title (str | None): Optional title of the plot
    """
    # Preview images (https://www.tensorflow.org/tutorials/load_data/images)
    plt.figure(figsize=(10, 20))
    for images, targets in dataset.take(1):
        if title:
            plt.suptitle(title)
        for i in range(0, 8, 2):
            plt.subplot(4, 2, i + 1)
            plt.imshow(images[i].numpy())
            plt.axis("off")

            plt.subplot(4, 2, i + 2)
            plt.imshow(targets[i].numpy())
            plt.axis("off")


def get_tuple_from_dataset(dataset: tf.data.Dataset) -> Tuple[Any, Any]:
    """Return the first (image, target) from a dataset

    Args:
        dataset (tf.data.Dataset): Dataset containing images

    Returns:
        Any: Image
    """
    for images, targets in dataset.take(1):
        return images[0], targets[0]
