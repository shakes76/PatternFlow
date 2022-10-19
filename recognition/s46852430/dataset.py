# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 17:57:39 2022

@author: eudre

"""
#importing the libraries
import os 
import numpy as np
import glob
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf

train_path ='C:\\Users\\eudre\\test\\ISIC-2017_Training_Data'
mask_path ='C:\\Users\\eudre\\test\\ISIC-2017_Training_Part1_GroundTruth'

H = 256
W = 256

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
# Use to delete superpixel image
def delete_super(path):
    os.chdir(train_path)
    for fname in os.listdir(train_path):
        if fname.endswith('superpixels.png') & fname.endswith('.csv'):
            os.remove(fname)

def load_data(train_path, mask_path):
    images = sorted(glob(os.path.join(train_path, "*.jpg")))
    masks = sorted(glob(os.path.join(mask_path, "*.jpg")))
    
    test_size = int(len(images) * 0.2)
    
    train_x, valid_x = train_test_split(images, test_size=test_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=test_size, random_state=42)

    train_x, test_x = train_test_split(train_x, test_size=test_size, random_state=42)
    train_y, test_y = train_test_split(train_y, test_size=test_size, random_state=42)
   
    return (train_x, train_y), (valid_x, valid_y), (test_x, test_y)

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)  ## (H, W, 3)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)
    return x                                ## (256, 256, 3)

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (H, W)
    x = cv2.resize(x, (W, H))
    x = x/255.0
    x = x.astype(np.float32)                    ## (256, 256)
    x = np.expand_dims(x, axis=-1)              ## (256, 256, 1)
    return x

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y

    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([H, W, 3])
    y.set_shape([H, W, 1])
    return x, y

def tf_dataset(X, Y, batch):
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset



