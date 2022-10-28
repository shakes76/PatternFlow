# ## Import the libraries
from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ## Loading the images

#Import the images and perform transformation on images

transformed_X = 256
transformed_Y = 256
def load_images(path):
    image_list = []
    for fi in os.listdir(path):
    #print(fi)
        if fi.endswith(".csv") or "superpixels" in fi:
            continue
        img = cv2.imread(os.path.join(path, fi),cv2.IMREAD_COLOR)
        img = cv2.resize(img,(transformed_Y,transformed_X))
        img = img / 255.0
        img = img.astype(np.float32)
        image_list.append(img)
    image_list = np.array(image_list)
    return image_list


# ## Load mask images

#Import the mask dataset
def load_masks(path):
    masks_list = []
    for fi in os.listdir(path):
    #print(fi)
        if fi.endswith(".csv") or "superpixels" in fi:
            continue
        mask = cv2.imread(os.path.join(path, fi),cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,(transformed_Y,transformed_X),interpolation = cv2.INTER_NEAREST)
        mask = mask / 255.0
        mask = mask.astype(np.float32)
        masks_list.append(mask)
    masks_list = np.array(masks_list)
    return masks_list


# ## Training data

X_train = load_images("ISIC-2017_Training_Data/")
masks_train_images = load_masks("ISIC-2017_Training_Part1_GroundTruth/")


# ## Validation data
x_validate = load_images("ISIC-2017_Validation_Data/")
masks_valid_images = load_masks("ISIC-2017_Validation_Part1_GroundTruth")


x_test = load_images("ISIC-2017_Test_v2_Data/")
masks_test_images = load_images("ISIC-2017_Test_v2_Part1_GroundTruth")

