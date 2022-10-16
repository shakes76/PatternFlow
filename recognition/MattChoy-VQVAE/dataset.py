import constants

import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

dataset_fp = os.path.join(os.getcwd(), "data")

train_ds = image_dataset_from_directory(
    os.path.join(dataset_fp, "train"),
    batch_size=constants.batch_size,
    image_size=constants.raw_image_size,
    seed=constants.dataset_seed,
    label_mode = None
)

test_ds = image_dataset_from_directory(
    os.path.join(dataset_fp, "test"),
    batch_size=constants.batch_size,
    image_size=constants.raw_image_size,
    seed=constants.dataset_seed,
    label_mode = None
)

train_scaled_ds = image_dataset_from_directory(
    os.path.join(dataset_fp, "train"),
    batch_size=constants.batch_size,
    image_size=constants.image_size,
    seed=constants.dataset_seed,
    label_mode = None
)

def scaling(input_image):
    input_image = input_image / 255.0
    input_image = tf.image.rgb_to_grayscale(input_image)
    input_image = tf.image.resize(input_image, constants.image_size)
    return input_image

train_ds = train_ds.map(scaling)
test_ds = test_ds.map(scaling)

def downscale(input_image):
    input_image = input_image / 255.0
    input_image = tf.image.rgb_to_grayscale(input_image)
    input_image = tf.image.resize(input_image, constants.image_size)
    input_image -= 0.5
    return input_image

train_scaled_ds = train_scaled_ds.map(downscale)
