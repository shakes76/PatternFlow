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
import gc
#%%

train_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2'
seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2'
#%%
def datasplitter(load_train_path,load_seg_path,save_train_path,save_seg_path,resize=1):
    
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
    
    
    # clean the train image save path
    for root, dirs, files in os.walk(save_train_path):
        for file in files:
            os.remove(os.path.join(root, file))
            
    # clean the train mask image save path
    for root, dirs, files in os.walk(save_seg_path):
        for file in files:
            os.remove(os.path.join(root, file))

    for x in X_train:
        
        img_file = glob.glob(os.path.join(load_train_path,x[:-4]+"*"))
        x_img = cv2.imread(img_file[0])
        
        if resize == 1:
            img_w = int(x_img.shape[1] * 1)
            img_h = int(x_img.shape[0] * 1)
        else:
            if 750 <=x_img.shape[1]:
                img_w = int(x_img.shape[1] * resize)
            elif 750 > x_img.shape[1]:
                img_w = int(x_img.shape[1] * 1)
                
            if 750 <=x_img.shape[0]:
                img_h = int(x_img.shape[0] * resize)
            elif 750 > x_img.shape[0]:
                img_h = int(x_img.shape[0] * 1)
            
        x_img = cv2.resize(x_img,(img_w,img_h))
        cv2.imwrite(os.path.join(save_train_path,x),x_img)
        
        seg_file = glob.glob(os.path.join(load_seg_path,x[:-4]+"*"))
        x_seg = cv2.imread(seg_file[0])

        if resize == 1:
            seg_w = int(x_seg.shape[1] * 1)
            seg_h = int(x_seg.shape[0] * 1)
        else:
            if 750 <=x_seg.shape[1]:
                seg_w = int(x_seg.shape[1] * resize)
            elif 750 > x_seg.shape[1]:
                seg_w = int(x_seg.shape[1] * 1)
                
            if 750 <=x_seg.shape[0]:
                seg_h = int(x_seg.shape[0] * resize)
            elif 750 > x_seg.shape[0]:
                seg_h = int(x_seg.shape[0] * 1)

        x_img = cv2.resize(x_seg,(seg_w,seg_h))
        cv2.imwrite(os.path.join(save_seg_path,seg_file[0][-29:]),x_seg)
#%%    
    
    
train_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/train_img'
train_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/train_seg'
val_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/val_img'
val_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/val_seg'
test_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/test_img'
test_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/test_seg'

# Train Set
datasplitter(train_path,seg_path,train_img_path,train_seg_path,resize=0.25)

# Validation Set
datasplitter(train_path,seg_path,val_img_path,val_seg_path,resize=0.25)

# Test Set
datasplitter(train_path,seg_path,test_img_path,test_seg_path,resize=0.25)

print('Done')