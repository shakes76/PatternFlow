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

from constants import IMG_ORIG_SIZE, IMG_DOWN_SIZE, BATCH_SIZE


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
    testing and training data sets.

    Args:
        data_path (str): path to the folder containing the images

    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Tuple of train and test
        datasets (train, test)
    """
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    train_ds = image_dataset_from_directory(
        train_path,
        image_size=IMG_ORIG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
    )

    test_ds = image_dataset_from_directory(
        test_path,
        image_size=IMG_ORIG_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
    )

    normalisation_layer = keras.layers.Rescaling(1./255)  # Normalise

    norm_train_ds = train_ds.map(lambda x, y: (normalisation_layer(x), y))
    norm_test_ds = test_ds.map(lambda x, y: (normalisation_layer(x), y))

    return norm_train_ds, norm_test_ds


def downsample_data(
    dataset: tf.data.Dataset,
    resulting_size: tuple[int, int] = IMG_DOWN_SIZE,
) -> Any:
    """Return dataset with all images downsized to resulting_size

    Args:
        dataset (tf.data.Dataset): dataset containing images to downsize.
        resulting_size tuple[int, int]: The image size to downsample towards.
            Defaults to (64, 60).

    Returns:
        Any: Downsized images in a dataset
    """
    return dataset.map(
        lambda x, y: (tf.image.resize(x, resulting_size, method="bicubic"), y)
    )


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
    return tf.image.resize(image, resulting_size, method="bicubic")


def preview_data(dataset: tf.data.Dataset) -> None:
    """Construct a matplotlib figure to preview some given image dataset

    Args:
        dataset (tf.data.Dataset): Dataset to preview
    """
    # Preview images (https://www.tensorflow.org/tutorials/load_data/images)
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(labels.numpy()[i])
            plt.axis("off")
