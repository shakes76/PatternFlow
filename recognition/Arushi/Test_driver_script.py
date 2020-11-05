"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# =============================================================================
Author: Arushi Mahajan
Student Number: 45755833
Copyright: Copyright 2020, UNet with ISIC dataset
Credits: Arushi Mahajan, Shakes and Team
License: COMP3710
Version: 1.0.1
Maintainer: Arushi Mahjan
Email: arushi.mahajan@uqconnect.edu.au
Status: Dev
Date Created: 31/10/2020
Date Modified: 05/11/2020
Description: Test driver script that calls and runs the algorithm
# =============================================================================
"""


# calling the ISIC_dataset_with_UNET file
get_ipython().run_line_magic('run', 'ISIC_dataset_with_UNET.py')

# Import all the necessary libraries
import os 
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from keras import backend as K

from tqdm import tqdm_notebook, tnrange
from itertools import chain

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

# initialising the compression dimensions
img_width = 256
img_height = 256
border = 5

# asking the user to enter the path of training input and training groundtruth folder
train_input_path = input("Enter the Location of Training_Input folder: ")
train_groundtruth_path = input("Enter the Location of Training_GroundTruth folder: ")

def main():
    """This function will automatically call all the functions and produce the final result"""
    isic_features, isic_labels = load_dataset(train_input_path, train_groundtruth_path)    
    isic_features_sort, isic_labels_sort = sorting_labels(isic_features, isic_labels)
    X_isic_train = load_features(train_input_path+"/", isic_features_sort)       
    y_isic_train = load_labels(train_groundtruth_path+"/",isic_labels_sort)
    X_train, X_test, y_train, y_test, X_val, y_val = split_datatset(X_isic_train, y_isic_train)
    y_train_encode, y_test_encode, y_val_encode = encoding(y_train,y_test,y_val)
    input_img = Input((img_height, img_width, 1), name = 'img')
    model = get_unet(input_img, n_filters = 16, dropout = 0.05, batchnorm = True) # generating u-net model
    model.compile(optimizer = Adam(), loss = dice_loss, metrics = ["accuracy",dice_coeffient]) # compiling the model with Adam optimizer and dice loss
    model.summary() # model summary
    callbacks = [EarlyStopping(patience = 10, verbose = 1), ReduceLROnPlateau(factor = 0.1, patience = 5, min_lr = 0.00001, verbose = 1), ModelCheckpoint('ISIC_model.h5', verbose = 1, save_best_only = True, save_weights_only = True)] # initializing callback to choose the best model
   
    results = model.fit(X_train, y_train_encode, batch_size = 32, epochs = 60, callbacks = callbacks, validation_data = (X_val, y_val_encode)) # model fitting
    test_preds_reshape = best_model(model,X_test,y_test,y_test_encode) # loading the best model
    
    lossPlot(results) # plotting loss plot
    accuracyPlot(results) # plotting accuracy plot
    plot_ISIC(X_test,y_test,test_preds_reshape) # plotting output

# calling the main function 
main()






