import tensorflow as tf
import os
import cv2

def load_data():
    # Expects keras_png_slices_data folder to be in this directory, unzipped
    DIR = "./keras_png_slices_data/keras_png_slices"
    CAT = ["test", "train", "validate", "seg_test", "seg_train", "seg_validate"]

    raw_data = {
        "test": [],
        "train": [],
        "validate": [],
        "seg_test": [],
        "seg_train": [],
        "seg_validate": [],
    }

    for cat in CAT:
      path = DIR + "_" + cat
      for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        raw_data[cat].append(img_array)

    data = {}
    data["test"] = tf.cast(tf.convert_to_tensor(raw_data["test"]), tf.float32) / 255.0
    data["train"] = tf.cast(tf.convert_to_tensor(raw_data["train"]), tf.float32)
    data["validate"] = tf.cast(tf.convert_to_tensor(raw_data["validate"]), tf.float32) / 255.0

    return data
