import glob
import tensorflow as tf
import numpy as np
from PIL import Image
from config import train_dir, test_dir, valid_dir

# files = glob.glob(r"C:\Users\s4512925\COMP3710_data\keras_png_slices_train\*.png")
# train_x = np.asarray([np.asarray(Image.open(f)) / 255 for f in files])
# train_x = np.expand_dims(train_x, axis=-1)
# train_dataset = tf.data.Dataset.from_tensor_slices(train_x).batch(batch_size)

img_size = 256
# Load images. NOTE: image_dataset_from_directory is only available in tf 2.3+
img_height = img_size
img_width = img_size

test_x = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    shuffle=False,
    label_mode=None,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    )
# test_x = test_x.unbatch()

train_x = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    shuffle=False,
    label_mode=None,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    )
# train_x = train_x.unbatch()

valid_x = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir,
    shuffle=False,
    label_mode=None,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    )
# valid_x = valid_x.unbatch()