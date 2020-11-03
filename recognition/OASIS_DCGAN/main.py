'''
OASIS DCGAN

@author Peter Ngo
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Reshape, Conv2D, Conv2DTranspose, Flatten, Dense
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Model


import matplotlib.pyplot as plt
import numpy as np
import time
from IPython import display
import PIL
import modules.layers as layers

# Google Colab Module Settings
#import sys
#sys.path.append('local_modules')
#from google.colab import drive
#drive.mount('/content/drive')

# Import the data
list_ds = tf.data.Dataset.list_files('/content/drive/My Drive/Datasets/keras_png_slices_data/keras_png_slices_train/*')
print(len(list_ds))

# Reads an image from a file, decodes it into a dense tensor, and resizes it
# to a fixed shape.
def parse_image(filename):

  image = tf.io.read_file(filename)
  image = tf.image.decode_png(image, channels = 1)
  image = tf.image.convert_image_dtype(image, tf.float32)
  image = tf.image.resize(image, [256, 256])

  return image

# Map over the image dataset.
training_image_ds = list_ds.map(parse_image).batch(32).shuffle(10000)

