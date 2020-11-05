#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import sys
import pandas as pd
from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split


def loaddata(input_PATH,input_ids,truth_PATH,truth_ids):
    """
    load data, resize to 256,256
    split data set
    normalize data
    set images data to 0-1 label
    return; three dataset and their x and y
    
    """
    image_size=[256,256]
    # data box to contain data later
    X_input = np.zeros((len(input_ids), image_size[0],image_size[1],3), dtype=np.uint8)
    Y_input = np.zeros((len(input_ids), image_size[0],image_size[1],1), dtype=np.uint8)
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(input_ids), total=len(input_ids)):
        img = imread(input_PATH + id_ )   
        img = resize(img, (image_size[0], image_size[1]), mode='constant', preserve_range=True)
        #img = np.expand_dims(img,-1)
        X_input[n] = img
    sys.stdout.flush()
    for n, id_ in tqdm(enumerate(truth_ids), total=len(truth_ids)):
        label = imread(truth_PATH + id_ )
        label = resize(label, (image_size[0], image_size[1]), mode='constant', preserve_range=True)
        label = np.expand_dims(label,-1)
        Y_input[n] = label
    #split dataset    
    X_train, X_test, y_train,y_test = train_test_split(X_input,Y_input, test_size=0.10)
    X_train, X_val, y_train,y_val = train_test_split(X_train,y_train, test_size=0.20)
    
    X_train=X_train/255
    X_val=X_val/255
    X_test=X_test/255
    y_train=y_train/255
    y_val=y_val/255
    y_test=y_test/255
    # label data
    y_train[y_train>0.5]=1
    y_train[y_train<=0.5]=0
    y_val[y_val>0.5]=1
    y_val[y_val<=0.5]=0
    y_test[y_test>0.5]=1
    y_test[y_test<=0.5]=0
    
    
    return X_train,X_val,X_test,y_train,y_val,y_test

