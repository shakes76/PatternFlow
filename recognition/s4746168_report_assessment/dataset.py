# import tensorflow as tf
# import tf.keras
import os
import numpy as np
from PIL import Image
# import torch

X_train_path = ".../Data_Files/ISIC-2017_Training_Data"
Y_train_path = ".../Data_Files/ISIC-2017_Training_Part1_GroundTruth"

X_validate_path = ".../Data_Files/ISIC-2017_Validation_Data"
Y_validate_path = ".../Data_Files/ISIC-2017_Validation_Part1_GroundTruth"

X_test_path = ".../Data_Files/ISIC-2017_Test_v2_Data"
Y_test_path = ".../Data_Files/ISIC-2017_Test_v2_Part1_GroundTruth"


def load_x_images(path):
    X = []
    for fi in os.listdir(path):
        if fi.endswith(".csv") or "superpixels" in fi:
            continue
        x = Image.open(os.path.join(path, fi)).resize((256, 256))
        x = np.array(x)
        # print(x.shape)
        # exit(0);
        x = x / 255.0
        # print(x_train.shape)
        X.append(x)

    np.stack(X)

    X = np.array(X)
    # X = torch.tensor(X)

    return X


def load_y_images(path):
    Y = []
    for fi in os.listdir(path):
        if fi.endswith(".csv") or "superpixels" in fi:
            continue
        y = Image.open(os.path.join(path, fi)).resize((256, 256), Image.NEAREST)
        y = np.array(y)

        # print(x_train.shape)
        Y.append(y)

    Y = np.array(Y)
    Y = Y / 255
    # Y = torch.tensor(Y)
    Y = Y[..., None]
    return Y


print("Program starting")

X_train = load_x_images(X_train_path)

# for f in os.listdir(X_train_path):
#     if f.endswith(".csv") or "superpixels" in f:
#         continue
#     x_train = Image.open(os.path.join(X_train_path, f)).resize((256, 256))
#     x_train = np.array(x_train)
#     x_train = x_train / 255.0
#     # print(x_train.shape)
#     X_train.append(x_train)
#
# np.stack(X_train)
print("********* Data loading for X_training complete *********")

Y_train = load_y_images(Y_train_path)
# for f in os.listdir(Y_train_path):
#     if f.endswith(".csv") or "superpixels" in f:
#         continue
#     y_train = Image.open(os.path.join(Y_train_path, f)).resize((256, 256))
#     y_train = np.array(y_train)
#     # print(y_train.shape)
#     Y_train.append(y_train)
#
# np.stack(Y_train)

# print(Y_train)
# print(np.unique(Y_train))
# print(Y_train.shape)
#
# exit(0)
print("********* Data loading for Y_training complete *********")

X_validate = load_x_images(X_validate_path)
print("********* Data loading for X_validating complete *********")

Y_validate = load_y_images(Y_validate_path)
print("********* Data loading for Y_validating complete *********")

X_test = load_x_images(X_test_path)
print("********* Data loading for X_testing complete *********")

Y_test = load_y_images(Y_test_path)
print("********* Data loading for Y_testing complete *********")

print(X_train.shape)
# print(X_train)
# print(Y_train.shape)

print(X_validate.shape)
# print(X_validate)

print(X_test.shape)
# print(X_test)
