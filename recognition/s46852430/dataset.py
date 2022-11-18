# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:24:14 2022

@author: eudre
"""

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
import tensorflow as tf



H = 256
W = 256



def read_images(inputs, channel):
    inputs=tf.image.decode_png(inputs,channels=channel)
    inputs=tf.image.resize(inputs,[H,W])
    inputs=tf.round(inputs/255.0)
    inputs=tf.cast(inputs,tf.float32)
    return inputs

def load_data(training, groundtruth):
    
    # Map out the dataset
    training=tf.io.read_file(training)
    training=read_images(training, 3)  
    groundtruth=tf.io.read_file(groundtruth)
    groundtruth=read_images(groundtruth, 1)   
    return training, groundtruth

def spilt_data(images, masks):
    images = sorted(glob.glob(images))
    masks = sorted(glob.glob(masks))  
    all_dataset = tf.data.Dataset.from_tensor_slices((images,masks))
    all_dataset = all_dataset.shuffle(len(images), reshuffle_each_iteration = False)
    train = all_dataset.take(int(0.7*(len(images))))
    test = all_dataset.take(int(0.15*(len(images))))
    valid = all_dataset.take(int(0.15*(len(images))))
    
    return train, test, valid