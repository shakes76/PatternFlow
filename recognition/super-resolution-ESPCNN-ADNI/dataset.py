"""dataset.py

Contains functions for loading and preprocessing the ADNI-MRI data.
"""

import os
from typing import Tuple
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import image_dataset_from_directory


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

    image_size = (256, 240)
    batch_size = 32
    
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    train_ds = image_dataset_from_directory(
        train_path,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
    )

    test_ds = image_dataset_from_directory(
        test_path,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
    )

    normalisation_layer = keras.layers.Rescaling(1./255)

    norm_train_ds = train_ds.map(lambda x, y: (normalisation_layer(x), y))
    norm_test_ds = test_ds.map(lambda x, y: (normalisation_layer(x), y))

    return norm_train_ds, norm_test_ds


def preview_data(dataset: tf.data.Dataset) -> None:
    """Construct a matplotlib figure to preview some given dataset

    Args:
        dataset (tf.data.Dataset): Dataset to preview
    """
    # Preview images (https://www.tensorflow.org/tutorials/load_data/images)
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy())
            plt.title(labels.numpy()[i])
            plt.axis("off")
