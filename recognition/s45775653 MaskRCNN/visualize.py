# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:45:43 2021

@author: chris
"""
#%%

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
import cv2
import skimage.io as skio
import sys
import time
from PIL import Image, ImageDraw
import gc

#%%
ROOT_DIR = 'D:/Python/mask_rcnn_aktwelve/Mask_RCNN/'
assert os.path.exists(ROOT_DIR), 'ROOT_DIR does not exist. Did you forget to read the instructions above? ;)'

sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
from mrcnn import visualize
import mrcnn.model as modellib

#%%
train_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2'
train_gt_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2'
#%%

# Read Data
img_spec = '*.jpg'
seg_spec = '*.png'
train_img = skio.imread_collection(os.path.join(train_path,img_spec),conserve_memory=True)
seg_img = skio.imread_collection(os.path.join(train_gt_path,seg_spec),conserve_memory=True)

check_seg = seg_img[0]
#%%
# how many images
print(len(train_img))

# resolution distribution
all_res = []

for res in train_img:
    all_res.append(res.shape[0]*res.shape[1])

plt.hist(np.divide(all_res,1000000)); plt.xlabel('megapixels'); plt.ylabel('No. of Images')

#%%

# Visualize
plt.figure(figsize=(10,15))
plt.subplot(1,4,1);plt.imshow(train_img[0],cmap='gray')
plt.subplot(1,4,2);plt.imshow(seg_img[0],cmap='gray')
plt.subplot(1,4,3);plt.imshow(train_img[100],cmap='gray')
plt.subplot(1,4,4);plt.imshow(seg_img[100],cmap='gray')

#%%

# Verify that values != 0 OR 255 are not different class
img_0_chk = np.where((seg_img[0]>0) & (seg_img[0]<255),0,255)
img_100_chk = np.where((seg_img[100]>0) & (seg_img[100]<255),0,255)
img_200_chk = np.where((seg_img[200]>0) & (seg_img[200]<255),0,255)

plt.figure(figsize=(10,15));plt.title('Not 255 pixel values')
plt.subplot(1,3,1);plt.imshow(img_0_chk,cmap='gray')
plt.subplot(1,3,2);plt.imshow(img_100_chk,cmap='gray')
plt.subplot(1,3,3);plt.imshow(img_200_chk,cmap='gray')