import tensorflow as tf

import os

from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory

from IPython.display import display


dataset_url = "https://cloudstor.aarnet.edu.au/plus/s/L6bbssKhUoUdTSI/download"

data_path = keras.utils.get_file(origin=dataset_url, fname="ADNI", extract=True)
data_path = data_path[:-4]
train_path = os.path.join(data_path, "AD_NC/train")
test_path = os.path.join(data_path, "AD_NC/test")

batch_size = 8
image_size = (500, 500)

train_ds = image_dataset_from_directory(
    train_path,
    batch_size=batch_size,
    image_size=image_size,
    validation_split=0.2,
    subset="training",
    seed=1337,
    label_mode=None,
    crop_to_aspect_ratio=True
)

validation_ds = image_dataset_from_directory(
    train_path,
    batch_size=batch_size,
    image_size=image_size,
    validation_split=0.2,
    subset="validation",
    seed=1337,
    label_mode=None,
    crop_to_aspect_ratio=True
)

scale = lambda img : img / 255
# Scale from (0, 255) to (0, 1)
train_ds = train_ds.map(scale)
valid_ds = validation_ds.map(scale)

def input_downsample(input_image, initial_image_size, up_sample_factor=4):
  '''
  Downsample the images by a factor of up_sample_factor to generate low quality
  input images for the CNN
  '''
  input_image = input_process(input_image)
  output_size = initial_image_size // up_sample_factor
  return tf.image.resize(input_image, [output_size, output_size], method='area')

def input_process(input_image):
  '''
  Convert the images into the YUV colour space to make processing simpler for
  the GPU. Returns the greyscale channel of the YUV.
  '''
  input_image = tf.image.rgb_to_yuv(input_image)
  #split image into 3 subtensors along axis 3
  y, u, v = tf.split(input_image, 3, axis=3)
  #only return the y channel of the yuv (the greyscale)
  return y


train_ds = train_ds.map(lambda x: (input_downsample(x, image_size[0]), input_process(x)))
train_ds = train_ds.prefetch(buffer_size=32)

validation_ds = validation_ds.map(lambda x: (input_downsample(x, image_size[0]), input_process(x)))
