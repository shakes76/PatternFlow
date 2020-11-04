"""
Data imports handler. Exports data as tensorflow datasets for lazy loading
"""

import tensorflow as tf
from config import train_dir, test_dir, valid_dir

# recommended values for img_size and batch_size
img_size = 256
batch_size = 32
# Load images. NOTE: image_dataset_from_directory is only available in tf 2.3+
test_x = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    batch_size=batch_size,
    shuffle=False,
    label_mode=None,
    color_mode='grayscale',
    image_size=(img_size, img_size),
    )

train_x = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    batch_size=batch_size,
    shuffle=False,
    label_mode=None,
    color_mode='grayscale',
    image_size=(img_size, img_size),
    )

valid_x = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir,
    batch_size=batch_size,
    shuffle=False,
    label_mode=None,
    color_mode='grayscale',
    image_size=(img_size, img_size),
    )