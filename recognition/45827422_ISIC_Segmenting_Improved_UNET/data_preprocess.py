"""
     Author : Aravind Punugu
 Student ID : 45827422
       Date : 28 October 2021
GitHub Name : Tannishpage
"""

import os
import cv2
import tensorflow as tf
import numpy as np

def create_generator(path_to_data, path_to_gt, img_size):
    """
    Takes three variables, path_to_data and path_to_gt (groundtruth) and image size
    and loads images into numpy arrays, and returns them
    """
    # Opening and storing images in a list
    X_train = []
    Y_train = []
    for i, file in enumerate([f for f in os.listdir(path_to_data) if f.endswith(".jpg")]):
        # We traverse through all the images with file extension .jpg
        ximg = cv2.imread(os.path.join(path_to_data, file), 0) # We load the original image as is
        # for the mask we need to change the last bit of the name to _segmentation.png
        yimg = cv2.imread(os.path.join(path_to_gt, file.replace(".jpg", "_segmentation.png")), 0)
        # We resize the images
        ximg = cv2.resize(ximg, img_size)
        yimg = cv2.resize(yimg, img_size)
        # Normalize
        ximg = ximg/255.0
        yimg = yimg/255.0
        X_train.append(ximg)
        Y_train.append(yimg)

    Y_train = np.array(Y_train) # Convert to numpy array
    # Convert to one-hot encoding
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=2)

    return np.array(X_train), Y_train
