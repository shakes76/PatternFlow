import glob,os
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt

from prewitt import prewitt_filter

tf.enable_eager_execution()
data_dir = pathlib.Path('./resources/')
image_paths = list(data_dir.glob('./*'))
dataset = tf.data.Dataset.list_files(str(data_dir/'*'))

def load_img(file_path):
  img = tf.io.read_file(file_path)
  img = tf.image.decode_jpeg(img, channels=1)
  img = tf.cast(img, tf.float32)
  return img

images = dataset.map(load_img) 
#apply the prewitt filter to all images in resources
filtered_imgs = images.map(prewitt_filter)
#reshape tensor for visualisation
filtered_imgs = filtered_imgs.map(tf.squeeze)

for img in filtered_imgs:
    plt.imshow(img, cmap="gray")
    plt.show()