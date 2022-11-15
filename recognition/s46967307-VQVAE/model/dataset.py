import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
import pickle

def load_data():
    """
    load_data
    expects oasis dataset to be located in ./keras_png_slices_data,
    loads this data and returns a dictionary of test, train and validation data.
    """
    data = {
        "test": [],
        "train": [],
        "validate": [],
        #"seg_test": [],
        #"seg_train": [],
        #"seg_validate": [],
    }

    # Check if data already loaded:
    if os.path.isfile("data.pkl"):
        print("Found data.pkl, using that")
        with open('data.pkl', 'rb') as f:
            loaded_dict = pickle.load(f)
        data = loaded_dict
    else :
        print("Couldn't find data.pkl. Loading from raw file may take a moment")

        # Expects keras_png_slices_data folder to be in this directory, unzipped
        DIR = "./keras_png_slices_data/keras_png_slices"
        CAT = ["test", "train", "validate"]

        for cat in CAT:
          path = DIR + "_" + cat
          for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            data[cat].append(img_array)

        # Save the raw data for future use
        with open('data.pkl', 'wb') as f:
            pickle.dump(data, f)

    data["test"] = data["test"]
    data["train"] = data["train"]
    data["validate"] = data["validate"]

    data["test"] = tf.cast(data["test"], tf.float32) / 255.0
    data["train"] = tf.cast(data["train"], tf.float32) / 255.0
    data["validate"] = tf.cast(data["validate"], tf.float32) / 255.0

    return data
