# -*- coding: utf-8 -*-
"""
Created on Oct 28, 2020

@author: s4542006, Md Abdul Bari
"""

import os
import numpy as np
import tqdm.notebook 
from skimage.transform import resize
from keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split


def load_image(image_path=None):
    """Resizes images and stores pixel values in a multidimensional array
    
    Parameter: 
        directory (str): path for the image source
        
    Return:
        a multi-dimensional array[[float]]: having pixel values of the images
    """
    if image_path == None:
        image_path = input()
    # resized image dimension
    IMG_HEIGHT = IMG_WIDTH = 256
    ids = next(os.walk(image_path))[2] 
    ids = sorted(ids)
    X = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    for n, id_ in tqdm.notebook.tqdm(enumerate(ids), total=len(ids)):
        # load input images
        img = load_img(image_path + "/" + id_, color_mode = "grayscale")
        x_img = img_to_array(img)
        x_img = resize(x_img, (IMG_HEIGHT, IMG_WIDTH, 1), mode = 'constant', 
                       preserve_range = True)
        # store images
        X[n] = x_img
    return X


def data_part_normal(test_ratio=0.10, val_ratio=0.20):
    """splits the whole dataset into test, validattion and test sets 
    and normalize pixel values.
    
    Parameters:
        test_ratio (float): data proportion for testing model performance
        val_ratio (float): data proportion for tuning model hyperparameters
    
    Return: 
        train-validation-test sets {dict}: partitioned and normalized data sets
    """
    input_path = input("\nProvide directory path for 'INPUT' images: ")
    gt_path = input("\nProvide directory path for 'GROUND TRUTH' images: ")
    print("\nMany Thanks :))")
    print("Now loading images to be followed by model training")
    print("This might take a bit longer (depending on your system)\n")
    # load input image data
    X = load_image(image_path=input_path)
    # load ground truth image data
    y = load_image(image_path=gt_path)
    # get  index up to which alidation set will be separated
    val_size = int(X.shape[0] * val_ratio)
    # partitioning test test from others
    X_tr_full, X_test_, y_tr_full, y_test_ = train_test_split(X, y, 
                                                          test_size=test_ratio,
                                                          random_state=42)
    # normalize the data value (0-1)
    X_test, y_test = X_test_ / 255.0, (y_test_ // 255).astype("uint8")
    X_tr_norm, y_tr_norm = X_tr_full / 255.0, (y_tr_full // 255).astype("uint8")
    # split the remaining images into train and validation sets
    X_val, X_train = X_tr_norm[:val_size], X_tr_norm[val_size:]
    y_val, y_train = y_tr_norm[:val_size], y_tr_norm[val_size:]
    return {"train":(X_train, y_train), "validation":(X_val, y_val),
            "test": (X_test, y_test)}  


if __name__ == "__main__":
    print("This module provides utility functions training and testing UNET",
          "and is not meant to be executed on its own.")