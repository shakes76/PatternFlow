from google.colab import drive
drive.mount('/content/drive')

import numpy as np


import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from glob import glob
from functools import partial

import tensorflow as tf

train_path = './drive/MyDrive/Colab Notebooks/keras_png_slices_data/keras_png_slices_train/*.png'
test_path = './drive/MyDrive/Colab Notebooks/keras_png_slices_data/keras_png_slices_test/*.png'
val_path = './drive/MyDrive/Colab Notebooks/keras_png_slices_data/keras_png_slices_validate/*.png'
seg_train_path = './drive/MyDrive/Colab Notebooks/keras_png_slices_data/keras_png_slices_seg_train/*.png'
seg_test_path = './drive/MyDrive/Colab Notebooks/keras_png_slices_data/keras_png_slices_seg_test/*.png'
seg_val_path = './drive/MyDrive/Colab Notebooks/keras_png_slices_data/keras_png_slices_seg_validate/*.png'


# Load dataset from google drive. Note this is colab version.
def load_ds_train():
    path = './drive/MyDrive/Colab Notebooks/keras_png_slices_data/keras_png_slices_train/'
    ds_train = tf.keras.utils.image_dataset_from_directory(
        path, label_mode=None,
        image_size=(64, 64), batch_size=32
    )
    return ds_train


def load_seg_ds_train():
    path = './drive/MyDrive/Colab Notebooks/keras_png_slices_data/keras_png_slices_seg_train/'
    ds_train = tf.keras.utils.image_dataset_from_directory(
        path, label_mode=None,
        image_size=(64, 64), batch_size=32
    )
    return ds_train

def log2(x):
    return int(np.log2(x))


# we use different batch size for different resolution, so larger image size
# could fit into GPU memory. The keys is image resolution in log2
batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 8, 8: 4, 9: 2, 10: 1}
# We adjust the train step accordingly
train_step_ratio = {k: batch_sizes[2] / v for k, v in batch_sizes.items()}

def resize_image(res, image):
    # only donwsampling, so use nearest neighbor that is faster to run
    image = tf.image.resize(
        image, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image


def create_dataloader(res):
    batch_size = batch_sizes[log2(res)]
    # NOTE: we unbatch the dataset so we can `batch()` it again with the `drop_remainder=True` option
    # since the model only supports a single batch size
    ds_train = load_ds_train()
    dl = ds_train.map(partial(resize_image, res), num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    dl = dl.shuffle(200).batch(batch_size, drop_remainder=True).prefetch(1).repeat()
    return dl


def plot_images(images, log2_res, fname=""):
    scales = {2: 0.5, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 8: 6, 9: 7, 10: 8}
    scale = scales[log2_res]

    grid_col = min(images.shape[0], int(32 // scale))
    grid_row = 1

    f, axarr = plt.subplots(
        grid_row, grid_col, figsize=(grid_col * scale, grid_row * scale)
    )

    for row in range(grid_row):
        ax = axarr if grid_row == 1 else axarr[row]
        for col in range(grid_col):
            ax[col].imshow(images[row * grid_col + col])
            ax[col].axis("off")
    plt.show()
    if fname:
        f.savefig(fname)