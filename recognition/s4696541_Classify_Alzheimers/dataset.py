"""
Data loading and preprocessing for alzheimers classification data.
"""
import math
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_datasets as tfds
import numpy as np

TRAIN_FILE_PATH = "<INSERT TRAIN DATA PATH>"
TEST_FILE_PATH = "<INSERT TEST DATA PATH>"
TEST_IMAGES = 21520
VALIDATION_SPLIT = (((TEST_IMAGES*0.1) // 20) * 20)/TEST_IMAGES

BATCH_SIZE = 32
IMAGE_DIM = 240
IMAGE_SIZE = (IMAGE_DIM, IMAGE_DIM)
SEED = 1337
PATCH_SIZE = 8
NUM_PATCHES = (IMAGE_DIM//PATCH_SIZE) ** 2
D_MODEL = (PATCH_SIZE**2) * 3

def training_dataset() -> tf.data.Dataset:
    dataset = keras.preprocessing.image_dataset_from_directory(TRAIN_FILE_PATH, labels='inferred', batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, seed=SEED, validation_split=VALIDATION_SPLIT, subset="training")
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def validation_dataset() -> tf.data.Dataset:
    dataset = keras.preprocessing.image_dataset_from_directory(TRAIN_FILE_PATH, labels='inferred', batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, seed=SEED, validation_split=VALIDATION_SPLIT, subset="validation")
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def testing_dataset() -> tf.data.Dataset:
    dataset = keras.preprocessing.image_dataset_from_directory(TEST_FILE_PATH, labels='inferred', batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def make_patch(image):
    batch_size = tf.shape(image)[0]
    patches = tf.image.extract_patches(
        images=image,
        sizes=[1, PATCH_SIZE, PATCH_SIZE, 1],
        strides=[1, PATCH_SIZE, PATCH_SIZE, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    patch_dims = patches.shape[-1]
    patches = tf.reshape(patches, [batch_size, NUM_PATCHES, patch_dims])

    return patches

import matplotlib.pyplot as plt

#Test that training dataset can properly get an image and create patches
if __name__ == "__main__":
    plt.figure(figsize=(10, 10))
    ds = training_dataset()
    images, label = next(iter(ds))
    n = int(math.sqrt(NUM_PATCHES))
    patches = make_patch(images)
    for j, patch in enumerate(patches[0]):
        plt.subplot(n, n, j + 1)
        patch_img = tf.reshape(patch, (PATCH_SIZE, PATCH_SIZE, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")

    plt.savefig(f"dataset_patching_example.png")