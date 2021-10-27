"""
The following code will take two folders as inputs namely the data folder and
the groundtruth folder, and split these into training, testing and
validation. Then create and return generators for those
"""

import sys
import os
import shutil
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import tensorflow as tf
import numpy as np

def create_generator(path_to_data, path_to_gt, img_size, nums):
    # Creating Train generators
    X_train = []
    Y_train = []
    for i, file in enumerate([f for f in os.listdir(path_to_data) if f.endswith(".jpg")]):
        ximg = cv2.imread(os.path.join(path_to_data, file), 0)
        yimg = cv2.imread(os.path.join(path_to_gt, file.replace(".jpg", "_segmentation.png")), 0)
        ximg = cv2.resize(ximg, img_size)
        yimg = cv2.resize(yimg, img_size)
        ximg = ximg/255.0
        yimg = yimg/255.0
        X_train.append(ximg)
        Y_train.append(yimg)

    Y_train = np.array(Y_train)
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=2)

    return np.array(X_train), Y_train
