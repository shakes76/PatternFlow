from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
import numpy as np

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


