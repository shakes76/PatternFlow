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

"""
    # Function to load the preprocessed images
    # This function only loads images into X type of variables 
    # i.e. the real time images  
"""


def load_x_images(path):
    X = []
    for fi in os.listdir(path):
        if fi.endswith(".csv") or "superpixels" in fi:
            continue

        x = Image.open(os.path.join(path, fi)).resize((256, 256))
        x = np.array(x)
        
        # Normalising the data

        x = x / 255.0

        # print(x_train.shape)
        X.append(x)

    np.stack(X)

    """
        # All the list are now converted to numpy array for final use
        # These array are then returned 
    """

    X = np.array(X)
    # X = torch.tensor(X)

    return X


"""
    # Function to load the preprocessed images
    # This function only loads images into Y type of variables 
    # i.e. the images with detection of cancer
    # Non real time images 
"""


def load_y_images(path):
    Y = []
    for fi in os.listdir(path):
        if fi.endswith(".csv") or "superpixels" in fi:
            continue
        y = Image.open(os.path.join(path, fi)).resize((256, 256), Image.NEAREST)
        y = np.array(y)

        # print(x_train.shape)
        Y.append(y)

    """
        # Data being converted to numpy array so that it can be finally used
        # These array are returned upon final functioning
    """
    Y = np.array(Y)

    # Normalising Data 

    Y = Y / 255
    
    # COnverted to proper classes 
    Y = Y[..., None]
    
    return Y


"""
    # Loading normal images for training the model
    # i.e. X_train is loaded with images 
"""

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


"""
    # Loading segmented images for training the model
    # i.e. Y_train is loaded with segmented images with cancer detected
"""

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

"""
    # Loading normal images for validating the model
    # i.e. X_validate is loaded with images
"""

X_validate = load_x_images(X_validate_path)
print("********* Data loading for X_validating complete *********")

"""
    # Loading segmented images for the validation of model
    # i.e. Y_validate is loaded with segmented images
"""

Y_validate = load_y_images(Y_validate_path)
print("********* Data loading for Y_validating complete *********")

"""
    # Loading normal images for testing the model
    # i.e. X_test is loaded with images
"""

X_test = load_x_images(X_test_path)
print("********* Data loading for X_testing complete *********")

"""
    # Loading segmented images for testing the model
    # i.e. Y_test is loaded with images
"""

Y_test = load_y_images(Y_test_path)
print("********* Data loading for Y_testing complete *********")

"""
    # Shapes have been printed 
    # to check the number of images and sizes of the variables
    # Hence checking the loading of dataset is done correctly  
"""

print(X_train.shape)
# print(X_train)
# print(Y_train.shape)

print(X_validate.shape)
# print(X_validate)

print(X_test.shape)
# print(X_test)
