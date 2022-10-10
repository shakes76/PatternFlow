import pathlib
import tensorflow as tf

import os
import math
import numpy as np

import matplotlib.pyplot as plt

CROP_SIZE = 300
BATCH_SIZE = 8
UPSCALE_FACTOR = 3
INPUT_SIZE = CROP_SIZE // UPSCALE_FACTOR

# import the data
def import_train_data():
    train_dir_str = r"C:\Users\galla\OneDrive\University\Year 3\Semester 2\COMP3710\Report\AD_NC\train\AD"
    train_dir = pathlib.Path(train_dir_str)
    return train_dir

def import_test_data():
    test_dir_str = r"C:\Users\galla\OneDrive\University\Year 3\Semester 2\COMP3710\Report\AD_NC\test\AD"
    test_dir = pathlib.Path(test_dir_str)
    return test_dir

# Creating training and validation datasets
def creating_train_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        import_train_data(),
        batch_size=BATCH_SIZE,
        image_size=(CROP_SIZE, CROP_SIZE),
        validation_split=0.2,
        subset="training",
        seed=1337,
        label_mode=None,
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        import_train_data(),
        batch_size=BATCH_SIZE,
        image_size=(CROP_SIZE, CROP_SIZE),
        validation_split=0.2,
        subset="validation",
        seed=1337,
        label_mode=None,
    )

    return train_ds, valid_ds

def creating_test_dataset():
    test_ds = tf.keras.utils.image_dataset_from_directory(
        import_test_data(),
        batch_size=BATCH_SIZE,
        seed=1337,
        label_mode=None,
    )
    return test_ds

# Rescaling
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image


# Scale from (0, 255) to (0, 1)
def mapping():
    train_ds_raw, valid_ds_raw = creating_train_datasets()

    train_ds = train_ds_raw.map(scaling)
    valid_ds = valid_ds_raw.map(scaling)

    return train_ds, valid_ds

train_ds, valid_ds = mapping()
test_ds = creating_test_dataset()


# #Visualising images
# for batch in train_ds.take(1):
#     for img in batch:
#         img_plot = plt.imshow(img)

# converting images from RGB to YUV from the low-resolution images
def process_input(input, input_size):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")

# converting images from RBG to YUB from the high-resolution images
def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y

def mapping_yuv(train_ds, valid_ds):
    train_ds = train_ds.map(
        lambda x: (process_input(x, INPUT_SIZE), process_target(x))
        )

    train_ds = train_ds.prefetch(buffer_size=32)

    valid_ds = valid_ds.map(
        lambda x: (process_input(x, INPUT_SIZE), process_target(x))
        )

    valid_ds = valid_ds.prefetch(buffer_size=32)

    return train_ds, valid_ds

train_yuv_ds, valid_yuv_ds = mapping_yuv(train_ds, valid_ds)

# Visualise input and target
# for batch in train_ds.take(1):
#     for img in batch[0]:
#         img_plot = plt.imshow(img)
#     for img in batch[1]:
#         img_plot = plt.imshow(img)
