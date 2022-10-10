import pathlib
import tensorflow as tf

import os
import math
import numpy as np

# from IPython.display import display

# import the data
def import_data():
    train_dir_str = r"C:\Users\galla\OneDrive\University\Year 3\Semester 2\COMP3710\Report\AD_NC\train\AD"
    train_dir = pathlib.Path(train_dir_str)
    return train_dir



# Creating training and validation datasets
def creating_datasets():

    crop_size = 300
    upscale_factor = 3
    input_size = crop_size // upscale_factor
    batch_size = 8

    train_ds = tf.keras.utils.image_dataset_from_directory(
        import_data(),
        batch_size=batch_size,
        image_size=(crop_size, crop_size),
        validation_split=0.2,
        subset="training",
        seed=1337,
        label_mode=None,
    )

    valid_ds = tf.keras.utils.image_dataset_from_directory(
        import_data(),
        batch_size=batch_size,
        image_size=(crop_size, crop_size),
        validation_split=0.2,
        subset="validation",
        seed=1337,
        label_mode=None,
    )

    return train_ds, valid_ds

# Rescaling
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image


# Scale from (0, 255) to (0, 1)
def mapping():
    train_ds_raw, valid_ds_raw = creating_datasets()

    train_ds = train_ds_raw.map(scaling)
    valid_ds = valid_ds_raw.map(scaling)

    return train_ds, valid_ds

train_ds, valid_ds = mapping()


#Visualising images
# for batch in train_ds.take(1):
#     for img in batch:
#         display(tf.keras.utils.array_to_img(img))