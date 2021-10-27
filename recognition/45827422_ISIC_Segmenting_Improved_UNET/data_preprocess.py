"""
The following code will take two folders as inputs namely the data folder and
the groundtruth folder, and split these into training, testing and
validation. Then create and return generators for those
"""

import sys
import os
import shutil
from tensorflow import keras as kr
from tensorflow.keras import preprocessing as krp

def create_generators(train, val, test, path_to_data, path_to_groundtruth):
    # Creating Train generators
    pass

def move_files(files, from_dest, destination):
    for file in files:
        shutil.copy(os.path.join(from_dest, file), destination)

def process_data_folders(path_to_data):
    data_files = [f for f in os.listdir(path_to_data) if f.endswith(".jpg")]
    print(data_files)
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
home = "/home/tannishpage/Documents/COMP3710_DATA"
train, val, test = process_data_folders("/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1-2_Training_Input_x2/")

move_files(train[0], "/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1-2_Training_Input_x2/", os.path.join(home, "train/data/images"))
move_files(train[1], "/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1_Training_GroundTruth_x2/", os.path.join(home, "train/groundtruth/images"))

move_files(val[0], "/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1-2_Training_Input_x2/", os.path.join(home, "val/data/images"))
move_files(val[1], "/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1_Training_GroundTruth_x2/", os.path.join(home, "val/groundtruth/images"))

move_files(test[0], "/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1-2_Training_Input_x2/", os.path.join(home, "test/data/images"))
move_files(test[1], "/home/tannishpage/Documents/COMP3710_DATA/ISIC2018_Task1_Training_GroundTruth_x2/", os.path.join(home, "test/groundtruth/images"))
