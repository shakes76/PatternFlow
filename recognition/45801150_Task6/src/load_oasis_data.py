import tensorflow as tf
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

dataset_prefix = "/home/tomdx/datasets/keras_png_slices_data/"
train_suffix = "keras_png_slices_train"
test_suffix = "keras_png_slices_test"
validation_suffix = "keras_png_slices_validate"


batch_size = 64
img_height = 256
img_width = 256

def get_data():

    train = []
    test = []
    val = []
    for root_name, dir_names, file_names in os.walk(dataset_prefix + train_suffix):
        file_names.sort()
        for file_name in file_names:
            img = img_to_array(load_img(root_name + "/" + file_name, color_mode="grayscale"))
            train.append(img)
        print(f"\rLoaded {i} train images", end='')
    print()
    i = 0
    for root_name, dir_names, file_names in os.walk(dataset_prefix + test_suffix):
        file_names.sort()
        for file_name in file_names:
            img = img_to_array(load_img(root_name + "/" + file_name, color_mode="grayscale"))
            test.append(img)
            print(f"\rLoaded {i} test images", end='')
    print()

    for root_name, dir_names, file_names in os.walk(dataset_prefix + validation_suffix):
        file_names.sort()
        for file_name in file_names:
            img = img_to_array(load_img(root_name + "/" + file_name, color_mode="grayscale"))
            val.append(img)
            print(f"\rLoaded {i} val images", end='')
    print()

    return np.array(train).squeeze(), np.array(test).squeeze(), np.array(val).squeeze()



