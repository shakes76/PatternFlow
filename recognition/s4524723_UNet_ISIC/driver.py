# Start by importing dependencies.

import pathlib
import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
import PIL.Image
import glob

# Import the model.
from comp3702_unet_isic import get_UNET_model

AUTOTUNE = tf.data.AUTOTUNE

print(tf.__version__)

# Some stuff to bug fix.

# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, 'Not enough GPU hardware devices available'
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
# os.getcwd()

# Import the data files from Google Drive to colab.

image_dir = './data/images/'
mask_dir = './data/masks/'
root_image_dir = pathlib.Path(image_dir)
root_mask_dir = pathlib.Path(mask_dir)

# See the first image in the directory
PIL.Image.open(list(root_image_dir.glob('*.jpg'))[0])
PIL.Image.open(list(root_mask_dir.glob('*.png'))[0])

# Count the number of images in both directories.
image_files = sorted(glob.glob(image_dir + '*.jpg'))
mask_files = sorted(glob.glob(mask_dir + '*.png'))
image_count = len(image_files)
mask_count = len(mask_files)
DATASET_SIZE = image_count
print('Number of images: ', image_count)
print('Number of masks: ', mask_count)
img_height, img_width = 128, 128


def decode_image(img):
    # Convert the compressed string to a 3D uint8 tensor
    img = tf.io.decode_jpeg(img, channels=3)
    # Resize the image to the desired size
    return tf.image.resize(img, [img_height, img_width])


def decode_mask(mask):
    mask = tf.io.decode_png(mask, channels=1)
    return tf.image.resize(mask, [img_height, img_width])


def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = decode_image(img)
    mask = tf.io.read_file(mask_path)
    mask = decode_mask(mask)
    return img, mask

train_ds = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
train_ds = train_ds.map(process_path)
for element in train_ds.take(1):
  print("Image shape: ", element[0].numpy().shape, "\n", "Mask shape: ", element[1].numpy().shape)
  print("First image tensor: ", element[0].numpy())
  print("First mask tensor: ", element[1].numpy())


def normalize_dataset(image, mask):
  normalization_layer = layers.Rescaling(1./255)
  image = normalization_layer(image)
  mask = normalization_layer(mask)
  return (image, mask)

normalized_ds = train_ds.map(normalize_dataset)
for element in normalized_ds.take(1):
  print("Image shape: ", element[0].numpy().shape, "\n", "Mask shape: ", element[1].numpy().shape)
  print("First image tensor: ", element[0].numpy())
  print("First mask tensor: ", element[1].numpy())
  print("Max pixel value in mask: ", np.max(element[1].numpy()))


# Batch the dataset

BATCH_SIZE = 64
BUFFER_SIZE = 1000

TRAIN_DATASET_SIZE = int(DATASET_SIZE * 0.6)
VALIDATION_DATASET_SIZE = int(DATASET_SIZE * 0.2)
TEST_DATASET_SIZE = int(DATASET_SIZE * 0.2)

normalized_ds = (
    normalized_ds
    .cache()
    .shuffle(BUFFER_SIZE)
    )

train_batches = (
    normalized_ds
    .take(TRAIN_DATASET_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_batches = (
     normalized_ds
    .skip(TRAIN_DATASET_SIZE)
    .take(VALIDATION_DATASET_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_batches = (
     normalized_ds
    .skip(TRAIN_DATASET_SIZE + VALIDATION_DATASET_SIZE)
    .take(TEST_DATASET_SIZE)
    .batch(BATCH_SIZE)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


model = get_UNET_model()

print(model.summary())