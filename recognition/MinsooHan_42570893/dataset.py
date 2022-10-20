import os

import tensorflow as tf
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPool2D, Concatenate, Dense, Input, UpSampling2D
from keras.models import Model

# Paths of the traing, vaildation, and test images of ISIC dataset.
training_images_path = 'E:\Uni\COMP3710\ISIC-2017_Training_Data'
training_images = os.listdir(training_images_path)

training_ground_truth_images_path = 'E:\Uni\COMP3710\ISIC-2017_Training_Part1_GroundTruth'
training_ground_truth_images = os.listdir(training_ground_truth_images_path)

vaildation_images_path = 'E:\Uni\COMP3710\ISIC-2017_Validation_Data'
vaildation_images = os.listdir(vaildation_images_path)

vaildation_ground_truth_images_path = 'E:\Uni\COMP3710\ISIC-2017_Validation_Part1_GroundTruth'
vaildation_ground_truth_images = os.listdir(vaildation_ground_truth_images_path)

test_images_path = 'E:\Uni\COMP3710\ISIC-2017_Test_v2_Data'
test_images = os.listdir(test_images_path)

test_ground_truth_images_path = 'E:\Uni\COMP3710\ISIC-2017_Test_v2_Part1_GroundTruth'
test_ground_truth_images = os.listdir(test_ground_truth_images_path)

