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

def data_augmenter(dataset,load_train_path,save_train_path,load_seg_path,save_seg_path,resize):
    
    # generate augmenters
    seq_h = iaa.Sequential([iaa.Fliplr(1)])
    seq_v = iaa.Sequential([iaa.Flipud(1)])

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
            if (700*700) <= x_img.shape[0]*x_img.shape[1]:
                img_w = int(x_img.shape[1] * resize)
                img_h = int(x_img.shape[0] * resize)
            
            # when more than max size, decrease resolution
            elif (700*700) > x_img.shape[0]*x_img.shape[1]:
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
        x_img = cv2.imread(img_file[0])
        
        if resize == 1:
            img_w = int(x_img.shape[1] * 1)
            img_h = int(x_img.shape[0] * 1)
        else:
            # 700*700 is the known max resolution local GPU can handle during training
            # when less than max size, retain resolution
            if (700*700) <= x_img.shape[0]*x_img.shape[1]:
                img_w = int(x_img.shape[1] * resize)
                img_h = int(x_img.shape[0] * resize)
            
            # when more than max size, decrease resolution
            elif (700*700) > x_img.shape[0]*x_img.shape[1]:
                img_w = int(x_img.shape[1] * 1)
                img_h = int(x_img.shape[0] * 1)
        
        # resize the image
        x_img = cv2.resize(x_img,(img_w,img_h))
            
        # flip image horizontally
        x_img_h = seq_h(image=x_img)
        
        # flip image vertically
        x_img_v = seq_v(image=x_img)
        
        # save images
        cv2.imwrite(os.path.join(save_seg_path,x),x_img)
        cv2.imwrite(os.path.join(save_seg_path,x[:-4]+'_h.png'),x_img_h)
        cv2.imwrite(os.path.join(save_seg_path,x[:-4]+'_v.png'),x_img_v)

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
    file_cleaner(save_seg_test_path # clean the train mask image save path   
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
        
#%%    
    
train_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2'
seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2'
train_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/train_img'
train_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/train_seg'
val_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/val_img'
val_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/val_seg'
test_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/test_img'
test_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/test_seg'
test_img_full_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/test_img_full'
test_seg_full_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/test_seg_full'


# Train Set
datasplitter(train_path,seg_path,
             train_img_path,train_seg_path,
             val_img_path,val_seg_path,
             test_img_path,test_seg_path, 
             test_img_full_path,test_seg_full_path, 
             resize=0.25)

print('Done')

#%%

# samp_train_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/samp_train_img'
# samp_train_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/samp_train_seg'
# samp_val_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/samp_val_img'
# samp_val_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/samp_val_seg'
# samp_test_img_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/samp_test_img'
# samp_test_seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/Data Split/samp_test_seg'


# # Train Set
# datasplitter(train_path,seg_path,
#              samp_train_img_path,samp_train_seg_path,
#              samp_val_img_path,samp_val_seg_path,
#              samp_test_img_path,samp_test_seg_path,
#              resize=0.25)