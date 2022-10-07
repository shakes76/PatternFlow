"""dataset.py

Contains functions for loading and preprocessing the ADNI-MRI data.
"""

import os
from typing import Tuple
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils import load_img
from keras.utils import array_to_img
from keras.utils import img_to_array
from keras.utils import image_dataset_from_directory
from IPython.display import display

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
    """Return train and test datasets from the given data directory

    Args:
        data_path (str): path to the folder containing the images
    
    Returns:
        Tuple[tf.data.Dataset, tf.data.Dataset]: Tuple of train and test
        datasets (train, test)
    """
    
    image_size = (256, 240)
    batch_size = 32
    
    train_ds = image_dataset_from_directory(
        data_path,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
    )
    
    test_ds = image_dataset_from_directory(
        data_path,
        image_size=image_size,
        batch_size=batch_size,
        color_mode="grayscale",
    )
    
    return train_ds, test_ds

def preview_data(dataset: tf.data.Dataset) -> None:
    """Construct a matplotlib figure to preview some of the given dataset

    Args:
        dataset (tf.data.Dataset): Dataset to preview
    """
    # Preview images (from https://www.tensorflow.org/tutorials/load_data/images)
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(labels.numpy()[i])
            plt.axis("off")
