"""
Setup:
1. Obtain a Personal Access Token from GitHub (requires GitHub Account)
    -> Navigate to https://github.com/settings/tokens to get a PAT
2. Clone ADNI dataset repo
    git clone https://{personal-access-token}@github.com/MattPChoy/ADNI-dataset.git data
"""

import tensorflow as tf

import os
import math
import numpy as np

from tensorflow.keras.preprocessing import image_dataset_from_directory

from constants import image_size, raw_image_size, batch_size, dataset_seed, n_channels
import matplotlib.pyplot as plt
dataset_fp = os.path.join(os.getcwd(), "data")

train_ds = image_dataset_from_directory(
    os.path.join(dataset_fp, "train"),
    batch_size=batch_size,
    image_size=raw_image_size,
    seed=dataset_seed,
    label_mode = None
)

test_ds = image_dataset_from_directory(
    os.path.join(dataset_fp, "test"),
    batch_size=batch_size,
    image_size=raw_image_size,
    seed=dataset_seed,
    label_mode = None
)

def scaling(input_image):
    input_image = input_image / 255.0
    input_image = tf.image.rgb_to_grayscale(input_image)
    input_image = tf.image.resize(input_image, image_size)
    return input_image

train_ds = train_ds.map(scaling)
test_ds = test_ds.map(scaling)

train_scaled_ds = image_dataset_from_directory(
    os.path.join(dataset_fp, "train"),
    batch_size=batch_size,
    image_size=image_size,
    seed=dataset_seed,
    label_mode = None
)

def downscale(input_image):
    input_image = input_image / 255.0
    input_image = tf.image.rgb_to_grayscale(input_image)
    input_image = tf.image.resize(input_image, image_size)
    input_image -= 0.5
    return input_image
train_scaled_ds = train_scaled_ds.map(downscale)
#
# n_batches = len(train_ds)
# n_samples = n_batches * batch_size
# ds_images = np.ndarray(shape=(len(train_ds) * batch_size, image_size[0], image_size[1], n_channels))
# idx=0
# for batch in train_ds:
#     for i in batch:
#         ds_images[idx] = i
#         idx += 1
#
# print(f"Dataset variance is {np.var(ds_images)/255.0}")
