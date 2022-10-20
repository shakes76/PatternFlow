import os
from tensorflow import keras
import numpy as np

SHAPE = (64,64)

def data_loader(dir_path, scale_flag):
    all_files = os.listdir(dir_path)
    image_list = []
    for i in all_files:
        image = keras.preprocessing.image.load_img(dir_path + "/" + i, grayscale = True, 
        target_size=SHAPE)
        if scale_flag:
          image_list.append(
                keras.preprocessing.image.img_to_array(image) / 255)
        else:
          image_list.append(
                keras.preprocessing.image.img_to_array(image))
    return np.array(image_list)