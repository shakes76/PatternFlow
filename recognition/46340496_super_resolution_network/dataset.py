import pathlib
import tensorflow as tf

import os
import math
import numpy as np

# from IPython.display import display

# import the data

train_dir_str = r"C:\Users\galla\OneDrive\University\Year 3\Semester 2\COMP3710\Report\AD_NC\train\AD"
train_dir = pathlib.Path(train_dir_str)

crop_size = 300
upscale_factor = 3
input_size = crop_size // upscale_factor
batch_size = 8

# Creating training and validation datasets

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
)

# Rescaling
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image


# Scale from (0, 255) to (0, 1)
train_ds = train_ds.map(scaling)
valid_ds = valid_ds.map(scaling)


#Visualising images
# for batch in train_ds.take(1):
#     for img in batch:
#         display(tf.keras.utils.array_to_img(img))