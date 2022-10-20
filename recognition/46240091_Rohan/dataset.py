from tensorflow import keras
import numpy as np
import glob

SHAPE = (64,64)

def data_loader(dir_path, scale_flag):
  """
  Loading and preprocessing data using keras image loader: 
  load_image and img_to_array and converting it to grayscale.
  """
  all_files = glob.glob(dir_path + "/*.png")
  image_list = []
  for i in all_files:
      image = keras.preprocessing.image.load_img(i, grayscale = True, 
      target_size=SHAPE)
      if scale_flag:
        image_list.append(
              keras.preprocessing.image.img_to_array(image) / 255)
      else:
        image_list.append(
              keras.preprocessing.image.img_to_array(image))
  return np.array(image_list)