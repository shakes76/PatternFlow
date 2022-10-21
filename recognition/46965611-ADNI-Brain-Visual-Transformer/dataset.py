"""
dataset.py

Data loader for loading and preprocessing data.

Author: Joshua Wang (Student No. 46965611)
Date Created: 11 Oct 2022
"""
import tensorflow as tf
from parameters import DATA_LOAD_PATH, IMAGE_SIZE, BATCH_SIZE


def load_data():
    """
    Loads the dataset that will be used into Tensorflow datasets.
    """
    train_data = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_LOAD_PATH + "/train", labels='inferred', label_mode='binary',
        image_size=[IMAGE_SIZE, IMAGE_SIZE], shuffle=True,
        batch_size=BATCH_SIZE, seed=8, class_names=['AD', 'NC']
    )

    test_data = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_LOAD_PATH + "/test", labels='inferred', label_mode='binary', 
        image_size=[IMAGE_SIZE, IMAGE_SIZE], shuffle=True,
        batch_size=BATCH_SIZE, seed=8, class_names=['AD', 'NC']
    )

    # Augment data
    normalize = tf.keras.layers.Normalization()
    flip = tf.keras.layers.RandomFlip(mode='horizontal', seed=8)
    rotate = tf.keras.layers.RandomRotation(factor=0.02, seed=8)
    zoom = tf.keras.layers.RandomZoom(height_factor=0.1, width_factor=0.1, seed=8)

    train_data = train_data.map(
        lambda x, y: (rotate(flip(zoom(normalize(x)))), y)
    )

    test_data = test_data.map(
        lambda x, y: (rotate(flip(zoom(normalize(x)))), y)
    )

    # Take half of the 9000 images from the test set as validation data
    validation_data = test_data.take(len(list(test_data))//2)

    # Use remaining 4500 images as test set
    test_data = test_data.skip(len(list(test_data))//2)

    return train_data, validation_data, test_data