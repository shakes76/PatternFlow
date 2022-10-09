from hashlib import sha1
from tkinter.tix import IMAGE
import tensorflow as tf
import matplotlib
import tf.keras
import os
import numpy as np
from PIL import Image
import torch

# DEFINING THE PATHS FOR LOADING DATA

X_train_path = ".../Data_Files/ISIC-2017_Training_Data"
Y_train_path = ".../Data_Files/ISIC-2017_Training_Part1_GroundTruth"

X_validate_path = ".../Data_Files/ISIC-2017_Validation_Data"
Y_validate_path = ".../Data_Files/ISIC-2017_Validation_Part1_GroundTruth"

X_test_path = ".../Data_Files/ISIC-2017_Test_v2_Data"
Y_test_path = ".../Data_Files/ISIC-2017_Test_v2_Part1_GroundTruth"

print("Program starting")


# X_train is loaded with images 


X_train = []
for f in os.listdir(X_train_path):
    if f.endswith(".csv") or "superpixels" in f:
        continue
    x_train = Image.open(os.path.join(X_train_path, f)).resize((256, 256))
    x_train = np.array(x_train)
    x_train = x_train / 255.0
    # print(x_train.shape)
    X_train.append(x_train)

np.stack(X_train)
print("********* Data loading for X_training complete *********")

# Y_train is loaded with images

Y_train = []
for f in os.listdir(Y_train_path):
    if f.endswith(".csv") or "superpixels" in f:
        continue
    y_train = Image.open(os.path.join(Y_train_path, f)).resize((256, 256), Image.NEAREST)

    y_train = np.array(y_train)
    y_train = y_train / 255
    # print(y_train.shape)
    Y_train.append(y_train)

np.stack(Y_train)

# X_validate is loaded with images

X_validate = []
for f in os.listdir(X_validate_path):
    if f.endswith(".csv") or "superpixels" in f:
        continue
    x_validate = Image.open(os.path.join(X_validate_path, f)).resize((256, 256))
    x_validate = np.array(x_validate)
    x_validate = x_validate / 255.0
    # print(x_train.shape)
    X_validate.append(x_validate)

np.stack(X_validate)
print("********* Data loading for X_validating complete *********")

# Y_validate is loaded with images

Y_validate = []
for f in os.listdir(Y_validate_path):
    if f.endswith(".csv") or "superpixels" in f:
        continue
    y_validate = Image.open(os.path.join(Y_validate_path, f)).resize((256, 256), Image.NEAREST)
    y_validate = np.array(y_validate)
    y_validate = y_validate / 255
    # print(y_train.shape)
    Y_validate.append(y_validate)

np.stack(Y_validate)
print("********* Data loading for Y_validating complete *********")

#X_test is loaded with data

X_test = []
for f in os.listdir(X_test_path):
    if f.endswith(".csv") or "superpixels" in f:
        continue
    x_test = Image.open(os.path.join(X_test_path, f)).resize((256, 256))
    x_test = np.array(x_test)
    x_test = x_test / 255.0
    # print(x_train.shape)
    X_test.append(x_test)

np.stack(X_test)
print("********* Data loading for X_testing complete *********")

#Y_test is loaded with data

Y_test = []
for f in os.listdir(Y_test_path):
    if f.endswith(".csv") or "superpixels" in f:
        continue
    y_test = Image.open(os.path.join(Y_test_path, f)).resize((256, 256), Image.NEAREST)
    y_test = np.array(y_test)
    y_test = y_test / 255
    print(x_train.shape)
    Y_test.append(y_test)

np.stack(Y_test)
print("********* Data loading for Y_testing complete *********")


# PRINTING THE SHAPES TO CHECK LOADING IS DONE CORRECTLY

print(X_train.shape)
# print(X_train)
print(Y_train.shape)

print(X_validate.shape)
print(Y_validate.shape)

print(X_test.shape)
print(Y_test.shape)
