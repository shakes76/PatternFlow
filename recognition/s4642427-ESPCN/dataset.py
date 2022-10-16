import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import array_to_img

'''
Base of the code to import and augment our datasets is taken from
https://keras.io/examples/vision/super_resolution_sub_pixel/
'''

# Our data is greyscale, but we can treat is as RGB and convert to YUV regardless
def process_input(input, input_size):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return tf.image.resize(y, [input_size, input_size], method="area")

def process_target(input):
    input = tf.image.rgb_to_yuv(input)
    last_dimension_axis = len(input.shape) - 1
    y, u, v = tf.split(input, 3, axis=last_dimension_axis)
    return y

def scale(input_image):
    input_image = input_image / 255.0
    return input_image

dataset_link = "https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download"
keras.utils.get_file(origin=dataset_link, fname="ADNI", extract=True)

crop_size = 300 # Will mess with the aspect ratio as our images are not square (??FIX)
batch_size = 8
upscale_factor = 4
input_size = crop_size // upscale_factor # 75

train_ds = image_dataset_from_directory(
    "/root/.keras/datasets/AD_NC",
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="training",
    seed=83,
    label_mode=None,
)

valid_ds = image_dataset_from_directory(
    "/root/.keras/datasets/AD_NC",
    batch_size=batch_size,
    image_size=(crop_size, crop_size),
    validation_split=0.2,
    subset="validation",
    seed=83,
    label_mode=None,
)

train_ds = train_ds.map(scale)
valid_ds = valid_ds.map(scale)

train_ds = train_ds.map(
    lambda x: (process_input(x, input_size), process_target(x))
)

valid_ds = valid_ds.map(
    lambda x: (process_input(x, input_size), process_target(x))
)