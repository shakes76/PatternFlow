from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from PIL import Image
from os import listdir
from os.path import isfile, join
import math


def get_filenames_from_dir(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f))]

x_names = get_filenames_from_dir("ISIC2018_Task1-2_Training_Input_x2")
y_names = get_filenames_from_dir("ISIC2018_Task1_Training_GroundTruth_x2")

# 15% of all the images are set aside as the test set
x_train_val, x_test, y_train_val, y_test = train_test_split(x_names, y_names, test_size=0.15, random_state=42)

# 17% of the non-test images are set aside as the validation set
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.17, random_state=42)

print(len(x_train))
print(len(x_test))
print(len(x_val))