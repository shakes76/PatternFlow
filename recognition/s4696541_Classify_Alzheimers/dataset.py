"""
Data loading and preprocessing for alzheimers classification data.
"""
import math
from matplotlib.figure import SubFigure
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

TRAIN_FILE_PATH = "../../../AD_NC/train"
TEST_FILE_PATH = "../../../AD_NC/test"

BATCH_SIZE = 64
IMAGE_DIM = 240
IMAGE_SIZE = (IMAGE_DIM, IMAGE_DIM)
SEED = 1337
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_DIM//PATCH_SIZE) ** 2

def training_dataset() -> tf.data.Dataset:
    dataset = keras.preprocessing.image_dataset_from_directory(TRAIN_FILE_PATH, labels='inferred', batch_size=BATCH_SIZE, validation_split=0.2, subset="training", seed=SEED, image_size=IMAGE_SIZE)
    dataset = dataset.map(lambda x,y: (make_patch(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def validation_dataset() -> tf.data.Dataset:
    dataset = keras.preprocessing.image_dataset_from_directory(TRAIN_FILE_PATH, labels='inferred', batch_size=BATCH_SIZE, validation_split=0.2, subset="validation", seed=SEED, image_size=IMAGE_SIZE)
    dataset = dataset.map(lambda x,y: (make_patch(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def testing_dataset() -> tf.data.Dataset:
    dataset = keras.preprocessing.image_dataset_from_directory(TEST_FILE_PATH, labels='inferred', batch_size=BATCH_SIZE, image_size=IMAGE_SIZE)
    dataset = dataset.map(lambda x,y: (make_patch(x), y), num_parallel_calls=tf.data.AUTOTUNE)
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
    patches = tf.reshape(patches, [batch_size, -1, patch_dims])

    return patches

import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.figure(figsize=(10, 10))
    ds = training_dataset()
    images, label = next(iter(ds))
    n = int(math.sqrt(images.shape[1]))
    for i, patch in enumerate(images[0]):
        plt.subplot(n, n, i + 1)
        patch_img = tf.reshape(patch, (PATCH_SIZE, PATCH_SIZE, 3))
        plt.imshow(patch_img.numpy().astype("uint8"))
        plt.axis("off")

    plt.savefig("dataset_patching_example.png")