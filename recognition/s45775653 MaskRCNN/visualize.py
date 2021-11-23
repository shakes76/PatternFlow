# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 10:45:43 2021

@author: christian burbon
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

#%%

# Define the dataset loader class

class ISICDataset(utils.Dataset):
    '''
    Genenerates keras based dataset for the images and masks
    '''
    
    def load_data(self,masks_dir,images_dir):
        self.masks_dir = masks_dir
        self.images_dir = images_dir
        # Define the filetypes to search for
        img_spec = '*.jpg' # train images
        
        # populate image dictionary
        img_names = []
        for imgfile in os.listdir(images_dir):
            if imgfile.endswith('.png') or imgfile.endswith('.jpg') or imgfile.endswith('.jpeg'):
                img_names.append(imgfile[:-4])
                
                image_id = imgfile[:-4]
                image_file_name = imgfile
                image_path = os.path.abspath(os.path.join(images_dir, image_file_name))
                
                self.add_image(source="isic",
                               image_id=image_id,
                               path=image_path)
                self.add_class("isic",1,"leison")
                self.load_image(imgfile)
   
    
    def load_mask(self):
        seg_spec = '*.png' # gt images
        seg_names = []
        instance_masks = []
        class_ids = []
        
        for segfile in os.listdir(self.masks_dir):
            if segfile.endswith('.png') or segfile.endswith('.jpg') or segfile.endswith('.jpeg'):
                seg_names.append(segfile[:-4])
                instance_masks.append(cv2.imread(os.path.join(self.masks_dir,segfile)))
                class_ids.append(1)
        mask = np.dstack(instance_masks)
        class_ids = np.array(class_ids,dtype=np.int32)
        
        return mask, class_ids

#%%


dataset_train = ISICDataset()
dataset_train.load_data(train_gt_path,train_path)
dataset_train.prepare()
        
#%%

annotations_test = {}

# populate image dictionary

isic_dict = {}

img_names = []
seg_names = []

# Create a list of train, and gt filenames to exclude any non image files and retain correct sequence
for imgfile in os.listdir(train_path):
    if imgfile.endswith('.png') or imgfile.endswith('.jpg') or imgfile.endswith('.jpeg'):
        img_names.append(imgfile)

for segfile in os.listdir(train_gt_path):
    if segfile.endswith('.png') or segfile.endswith('.jpg') or segfile.endswith('.jpeg'):
        seg_names.append(segfile)

seg_spec = '*.png'
seg_masks = skio.imread_collection(os.path.join(train_gt_path,seg_spec),conserve_memory=True)

for idx, img_name in enumerate(img_names):
    image_id = img_name[:-4]
    mask = seg_masks[idx]
    
    isic_dict[idx] = {
        'mask':segmentation,
        'iscrowd':0,
        'image_id': image,
        'category_id':1,
        }
    

    

# for seg in seg_names:
#     print(seg)
#     image = seg[:12]
#     segmentation = cv2.imread(os.path.join(train_gt_path,seg))
#     annotations_test[image].append(segmentation)



