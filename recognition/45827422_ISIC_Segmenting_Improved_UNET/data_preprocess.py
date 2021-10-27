"""
The following code will take two folders as inputs namely the data folder and
the groundtruth folder, and split these into training, testing and
validation. Then create and return generators for those
"""

import sys
import os
import shutil
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
import numpy as np

def create_generator(path_to_data, path_to_gt, img_size, nums):
    # Creating Train generators
    X_train = []
    Y_train = []
    for i, file in enumerate(os.listdir(path_to_data)):
        if i >= nums:
            break
        ximg = load_img(os.path.join(path_to_data, file), target_size=(128, 128), color_mode="grayscale")
        ximg = img_to_array(ximg)/255.0
        X_train.append(ximg)

    for i, file in enumerate(os.listdir(path_to_gt)):
        if i >= nums:
            break
        yimg = load_img(os.path.join(path_to_gt, file), target_size=(128, 128), color_mode="grayscale")
        yimg = img_to_array(yimg)/255.0
        Y_train.append(yimg)
    Y_train = np.array(Y_train)
    Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=2)

    return np.array(X_train), Y_train


def move_files(files, from_dest, destination):
    for file in files:
        shutil.copy(os.path.join(from_dest, file), destination)

def process_data_folders(path_to_data):
    data_files = [f for f in os.listdir(path_to_data) if f.endswith(".jpg")]
    num_files = len(data_files)

    # The groundtruth files are named similarly to the data_files.
    # ISIC_<Number>.jpg
    # ISIC_<Number>_segmentation.png
    # Split data into train, test, validation

    training_split = int(num_files*0.6)
    validation_split = training_split + int(num_files*0.1)
    testing_split = validation_split + int(num_files*0.3)

    training = data_files[0:training_split]
    training_gt = [t.replace(".jpg", "_segmentation.png") for t in training]
    validation = data_files[training_split:validation_split]
    validation_gt = [t.replace(".jpg", "_segmentation.png") for t in training]
    testing = data_files[validation_split:]
    testing_gt = [t.replace(".jpg", "_segmentation.png") for t in training]

    return (training, training_gt), (validation, validation_gt), (testing, testing_gt)

#/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1-2_Training_Input_x2/
#/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1_Training_GroundTruth_x2/
