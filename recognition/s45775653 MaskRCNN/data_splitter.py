# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 03:29:37 2021

@author: chris

Splits the original dataset into Train, Validation, and Test sets
Then rescales image to 25% of original for W or H is > 750
before using dataloader.py
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import glob
import cv2
import skimage.io as skio
import sys
import time
from PIL import Image, ImageDraw
import imgaug.augmenters as iaa
import gc

#%%

def file_cleaner(save_path):
    # clean the train image save path
    for root, dirs, files in os.walk(save_path):
        for file in files:
            os.remove(os.path.join(root, file))

def data_augmenter(dataset,load_train_path,save_train_path,load_seg_path,save_seg_path,max_res=700*700,resize):
    
    # generate augmenters
    seq_h = iaa.Sequential([iaa.Fliplr(1)])
    seq_v = iaa.Sequential([iaa.Flipud(1)])
    seq_hv = iaa.Sequential([iaa.Fliplr(1),iaa.Flipud(1)])

    for x in dataset:
        
        # DO training input
        img_file = glob.glob(os.path.join(load_train_path,x[:-4]+"*"))
        x_img = cv2.imread(img_file[0])
        
        if resize == 1:
            img_w = int(x_img.shape[1] * 1)
            img_h = int(x_img.shape[0] * 1)
        else:
            # 700*700 is the known max resolution local GPU can handle during training
            # when less than max size, retain resolution
            if max_res <= x_img.shape[0]*x_img.shape[1]:
                img_w = int(x_img.shape[1] * resize)
                img_h = int(x_img.shape[0] * resize)
            
            # when more than max size, decrease resolution
            elif max_res > x_img.shape[0]*x_img.shape[1]:
                img_w = int(x_img.shape[1] * 1)
                img_h = int(x_img.shape[0] * 1)
        
        # resize the image
        x_img = cv2.resize(x_img,(img_w,img_h))
            
        # flip image horizontally
        x_img_h = seq_h(image=x_img)
        
        # flip image vertically
        x_img_v = seq_v(image=x_img)
          
        # save images
        cv2.imwrite(os.path.join(save_train_path,x),x_img)
        cv2.imwrite(os.path.join(save_train_path,x[:-4]+'_h.jpg'),x_img_h)
        cv2.imwrite(os.path.join(save_train_path,x[:-4]+'_v.jpg'),x_img_v)
        
        # DO training ground truth
        img_file = glob.glob(os.path.join(load_seg_path,x[:-4]+"*"))
        seg_img = cv2.imread(img_file[0])
        
        if resize == 1:
            img_w = int(seg_img.shape[1] * 1)
            img_h = int(seg_img.shape[0] * 1)
        else:
            # 700*700 is the known max resolution local GPU can handle during training
            # when less than max size, retain resolution
            if max_res <= seg_img.shape[0]*seg_img.shape[1]:
                img_w = int(seg_img.shape[1] * resize)
                img_h = int(seg_img.shape[0] * resize)
            
            # when more than max size, decrease resolution
            elif max_res> seg_img.shape[0]*seg_img.shape[1]:
                img_w = int(seg_img.shape[1] * 1)
                img_h = int(seg_img.shape[0] * 1)
        
        # resize the image
        seg_img = cv2.resize(seg_img,(img_w,img_h))
            
        # flip image horizontally
        x_seg_h = seq_h(image=seg_img)
        
        # flip image vertically
        x_seg_v = seq_v(image=seg_img)

        # save images
        cv2.imwrite(os.path.join(save_seg_path,x),x_img)
        cv2.imwrite(os.path.join(save_seg_path,x[:-4]+'_h.png'),x_seg_h)
        cv2.imwrite(os.path.join(save_seg_path,x[:-4]+'_v.png'),x_seg_v)
#%%
def datasplitter(load_train_path,load_seg_path,
                 save_train_path,save_seg_train_path,
                 save_val_path,save_seg_val_path,
                 save_test_path,save_seg_test_path,
                 save_test_full_path,save_seg_test_full_path,
                 resize=1):
    
    from sklearn.model_selection import train_test_split
    
    train_img = []
    seg_img = []
    
    for filename in os.listdir(load_train_path):
    # get only picture file types
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            train_img.append(filename)
    
    # split dataset into train, test
    X_train, X_test = train_test_split(train_img, test_size=0.25, random_state=42)
    
    # split dataset into train, validation
    X_train, X_val= train_test_split(X_train, test_size=0.25, random_state=42)
    
    # Load filenames and save to folders
    
    print('Running Train')
    # TRAIN SET
    file_cleaner(save_train_path) # clean the train image save path
    file_cleaner(save_seg_train_path) # clean the train mask image save path
    data_augmenter(X_train,load_train_path,save_train_path,load_seg_path,save_seg_train_path,resize) # DO augmentation
    
    print('Running Val')   
    # VALIDATION SET
    file_cleaner(save_val_path) # clean the train image save path    
    file_cleaner(save_seg_val_path) # clean the train mask image save path
    data_augmenter(X_val,load_train_path,save_val_path,load_seg_path,save_seg_val_path,resize) # DO augmentation
    
    print('Running Test')
    # TEST SET
    
    file_cleaner(save_test_path) # clean the train image save path
    file_cleaner(save_seg_test_path) # clean the train mask image save path   
    data_augmenter(X_test,load_train_path,save_test_path,load_seg_path,save_seg_test_path,resize) # DO augmentation
        
    print('Running Test - Full Image')
    # TEST SET using full image resolution
    
    file_cleaner(save_test_full_path) # clean the train image save path
    file_cleaner(save_seg_test_full_path) # clean the train mask image save path

    for idx, x in enumerate(X_test):
        img_file = glob.glob(os.path.join(load_train_path,x[:-4]+"*"))
        x_img = cv2.imread(img_file[0])
        cv2.imwrite(os.path.join(save_test_full_path,x),x_img)
        
        seg_file = glob.glob(os.path.join(load_seg_path,x[:-4]+"*"))
        x_seg = cv2.imread(seg_file[0])
        cv2.imwrite(os.path.join(save_seg_test_full_path,seg_file[0][-29:]),x_seg)
        