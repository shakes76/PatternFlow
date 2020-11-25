# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:53:43 2020

@author: s4552215
"""


import UNET_Model_ISIC_SM as UM
import Image_Segment_Preprocess_ISIC_SM as ISP

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


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

#The Function to evaluate dice coefficient for overall test data sets
def dice_coeff(y_true, y_pred, smooth=1):
    intersect = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersect
    coeff_dice = K.mean((intersect + smooth) / (union + smooth), axis=0)
    return coeff_dice

# The Function to evaluate dice coefficient for each of the test segmentation and predicted segmentation images
def dice_coefflabelwise(y_true, y_pred, smooth=1):
    intersect = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersect
    coeff_dice_ind = (intersect + smooth) / (union + smooth)
    return coeff_dice_ind

# The Function to evaluate Dice Loss
def dice_loss(y_true, y_pred, smooth = 1):
    return 1 - dice_coeff(y_true, y_pred, smooth = 1)

# The Function to extract the ISIC test image file names
def img_test_fn(seg_path):
    ids_all = next(os.walk(seg_path))[2]
    ids_all_sort = sorted(ids_all)
    train_split = int(round(0.6*len(ids_all_sort),0))
    test_val_split = int(round(0.2*len(ids_all_sort),0))
    return ids_all_sort[train_split + test_val_split:train_split + 2*test_val_split]

# The Function to generate plots for lesion, actual segmentation and predicted segmentation images
def plot_ISIC_all(X, y, y_pred, ix=None):
    
    if ix is None:
        ix = random.randint(0, len(X))
    else:
        ix = ix

    
    # To Plot the original lesion grayscale images
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    ax[0].imshow(X[ix, ..., 0], cmap='gray')
    ax[0].contour(X[ix].squeeze(), colors='k', levels=[0.5])
    ax[0].set_title('Input Image')
    
    # To Plot the actual segmentation
    ax[1].imshow(y[ix,...,0],cmap='gray')
    ax[1].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    ax[1].set_title('Actual Segmentation')
    
    # To Plot the predicted segmentation
    ax[2].imshow(y_pred[ix,...,0],cmap='gray')
    ax[2].contour(y_pred[ix].squeeze(), colors='k', levels=[0.5])
    ax[2].set_title('Predicted Segmentation')
    plt.show()
    
# To Create the model, predict the test data, evaluate Dice Coefficients and display random lesion
# actual segmentation and predicted segmentation images

def mod_comp(img_path, seg_path, img_height, img_width):
    
    # To load the train, validation and test lesion images and segmentation images for ISIC data set
    XISIC_train, XISIC_val, XISIC_test = ISP.proc_img(img_path,img_width,img_height)
    YISIC_train, YISIC_val, YISIC_test = ISP.proc_seg(seg_path,img_width,img_height)
    
    # To transform the segmentation images into categorical data
    yISIC_train_cat, yISIC_val_cat, yISIC_test_cat = ISP.pix_cat(YISIC_train, YISIC_val, YISIC_test)
    
    #To Generate the U-NET model
    input_img = Input((img_height, img_width, 1), name='img')
    model = UM.unet_gen(input_img, n_fil=16, drop=0.05, batch=True)
    model.compile(optimizer=Adam(), loss=dice_loss, metrics=["accuracy",dice_coeff])
    print("Printing model summary\n")
    model.summary()
    
    # To Initialize the callback parameter to choose the best model best on the performance on the validation data set
    callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.00001, verbose=1),
    ModelCheckpoint('model-ISIC.h5', verbose=1, save_best_only=True, save_weights_only=True)]
    
    # To fit the U-NET model
    results = model.fit(XISIC_train, yISIC_train_cat, batch_size=32, epochs=30, callbacks=callbacks,validation_data=(XISIC_val, yISIC_val_cat))
    
    # To Plot the training and validation loss with respect to epochs
    plt.figure(figsize=(8, 8))
    plt.title("Dice Loss")
    plt.plot(results.history["loss"], label="training_loss")
    plt.plot(results.history["val_loss"], label="validation_loss")
    plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.legend();
    plt.show()
    
    # To Plot the training and validation accuracy with respect to epochs
    plt.figure(figsize=(8, 8))
    plt.title("Classification Accuracy")
    plt.plot(results.history["accuracy"], label="training_accuracy")
    plt.plot(results.history["val_accuracy"], label="validation_accuracy")
    plt.plot( np.argmax(results.history["val_accuracy"]), np.max(results.history["val_accuracy"]), marker="x", color="r", label="best model")
    plt.xlabel("Epochs")
    plt.legend();
    plt.show()
    
    # To Load the model with the best performance
    model.load_weights('model-ISIC.h5')
    
    # To Generate the predicted segmentation arrays of the ISIC Test data set
    ISIC_test_preds = model.predict(XISIC_test, verbose=1)
    
    print("The dice coefficient of the ISIC test data : ",dice_coeff(yISIC_test_cat,ISIC_test_preds).numpy())
    
    
    # To Create a dataframe for the individual dice coefficients of the segmentation images of the test data set
    test_data_dc = dice_coefflabelwise(yISIC_test_cat,ISIC_test_preds).numpy()
    test_data_img_files = np.array(img_test_fn(seg_path))
    test_fn_dc = pd.DataFrame({'ISIC_Test_Img': test_data_img_files, 'Dice_Coefficient': list(test_data_dc)}, columns=['ISIC_Test_Img', 'Dice_Coefficient'])
    print("Storing the individual dice coefficients of the segmentation images of the test data set\n")
    test_fn_dc.to_csv('Dice_Coefficients_Test.csv')
    
    # To generate random images for the predicted segmentations and to compare them with the actual segmentation of the lesion images
    ISIC_test_preds_max = np.argmax(ISIC_test_preds, axis = -1)
    n,h,w,g = YISIC_test.shape
    ISIC_test_preds_reshape = ISIC_test_preds_max.reshape(n,h,w,g)
    print("Comparing the predicted segmentation with the actual segmentation")
    plot_ISIC_all(XISIC_test,YISIC_test,ISIC_test_preds_reshape)
