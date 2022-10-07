import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow import keras

# Assumes Directory has a train and test folder

# Seed used for generating the training and validation dataset
seed = 123


def normalisation_data(ds):
    """
    Standardises input data then rescales it to be between 0 and 1.
    """
    normalisation_layer = tf.keras.layers.Normalization()
    scaling_layer = tf.keras.layers.Rescaling(1. / 255)
    norm_data = ds.map(lambda x, y: (normalisation_layer(x), y))
    clean_data = norm_data.map(lambda x, y: (scaling_layer(x), y))
    return clean_data


def load_train_data_no_val(path, height, width, batch_size):
    """
    Takes in a directory path, img height, width and batch size and returns normalised training data.
    Use function if no validation data needs to be generated from the given directory
    """
    train_data = tf.keras.utils.image_dataset_from_directory(path,
                                                             color_mode="rgb",
                                                             labels="inferred",
                                                             image_size=(height, width),
                                                             batch_size=batch_size)

    return normalisation_data(train_data)


def load_val_from_dir(path, height, width, batch_size):
    """
    Takes in a directory path, img height, width and batch size and returns normalised validation data.
    Use function if validation data is already given in a seperate directory
    """
    val_data = tf.keras.utils.image_dataset_from_directory(path,
                                                           color_mode="rgb",
                                                           labels="inferred",
                                                           image_size=(height, width),
                                                           batch_size=batch_size)

    return normalisation_data(val_data)


def load_train_data(path, height, width, batch_size, val_split):
    """
    Takes in a directory path, img height, width and batch size and returns normalised training data.
    Use this function for datasets that just have a train and test dataset, and a
    validation dataset needs to be generated.

    Val_split is the proportion of the data to be allocated to validation data.
    """

    train_data = tf.keras.utils.image_dataset_from_directory(path, validation_split=val_split,
                                                             subset="training",
                                                             seed=seed,
                                                             color_mode="rgb",
                                                             labels="inferred",
                                                             image_size=(height, width),
                                                             batch_size=batch_size)

    return normalisation_data(train_data)


def load_validation_data(path, height, width, batch_size, val_split):
    """
    Takes in a directory path, img height, width and batch size and returns normalised validation data.
    Use function if dataset just has train and test dataset and a validation set needs to be generated

    Val_split is the proportion of the data to be allocated to validation data.
    """
    validation_data = tf.keras.utils.image_dataset_from_directory(path, validation_split=val_split,
                                                                  subset="validation",
                                                                  seed=seed,
                                                                  color_mode="rgb",
                                                                  labels="inferred",
                                                                  image_size=(height, width),
                                                                  batch_size=batch_size)

    return normalisation_data(validation_data)


def load_test_data(path, height, width, batch_size):
    """
    Takes in a directory path, img height, width and batch size and returns normalised test data.
    """
    test_data = tf.keras.utils.image_dataset_from_directory(path,
                                                            color_mode="rgb",
                                                            labels="inferred",
                                                            image_size=(height, width),
                                                            batch_size=batch_size)

    return normalisation_data(test_data)


val_split = 0.2
img_height = 256
img_width = 256
batch_size = 32

PATH = os.getcwd()
print(PATH)

train_path = "AD_NC/train"
test_path = "AD_NC/test"
train_ds = load_train_data(train_path, img_height, img_width, batch_size, val_split)
val_ds = load_validation_data(train_path, img_height, img_width, batch_size, val_split)
test_ds = load_test_data(test_path, img_height, img_width, batch_size)

