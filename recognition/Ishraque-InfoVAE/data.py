import tensorflow as tf
from config import train_dir, test_dir, valid_dir
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
test_x = test_x.unbatch()

train_x = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    shuffle=False,
    label_mode=None,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    )
train_x = train_x.unbatch()

valid_x = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir,
    shuffle=False,
    label_mode=None,
    color_mode='grayscale',
    image_size=(img_height, img_width),
    )
valid_x = valid_x.unbatch()