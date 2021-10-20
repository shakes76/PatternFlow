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
import glob
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
seg_path = 'D:/UQ Data Science/Subjects/Semester 4/COMP3710 - Pattern Recognition/Final Report/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2'
#%%

# Read Data
img_spec = '*.jpg'
seg_spec = '*.png'
train_img = skio.imread_collection(os.path.join(train_path,img_spec),conserve_memory=True)
seg_img = skio.imread_collection(os.path.join(seg_path,seg_spec),conserve_memory=True)

check_seg = seg_img[0]


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
                # self.load_image(imgfile)
   
    
    def load_mask(self, image_id):
        # print(image_id)
        seg_spec = '*.png' # gt images
        seg_names = []
        instance_masks = []
        class_ids = []
        
        image_info = self.image_info[image_id]
        # print(image_info)
        filename = glob.glob(os.path.join(self.masks_dir,image_info['id']+"*"))
        # print(filename[0])
        mask_seg = cv2.imread(filename[0],0)
        # plt.imshow(mask_seg)
        # mask_seg = np.where(mask_seg>0,255,mask_seg)
        bool_mask = mask_seg>0
        instance_masks.append(bool_mask)
        class_ids.append(1)
        

        # for segfile in os.listdir(self.masks_dir):
        #     if segfile.endswith('.png') or segfile.endswith('.jpg') or segfile.endswith('.jpeg'):
        #         seg_names.append(segfile[:-4])
        #         mask_seg = cv2.imread(os.path.join(self.masks_dir,segfile))
        #         mask_seg = np.where(mask_seg>0,255,mask_seg)
        #         bool_mask = mask_seg>0
        #         instance_masks.append(bool_mask)
        #         class_ids.append(1)
        # mask = np.dstack(instance_masks)
        mask = instance_masks
        class_ids = np.array(class_ids,dtype=np.int32)
        
        return mask, class_ids

#%%
dataset_train = ISICDataset()
dataset_train.load_data(train_gt_path,train_path)
dataset_train.prepare()
#%%

dataset = dataset_train
image_ids = [0]
# image_ids = dataset.image_ids
for image_id in image_ids:
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, dataset.class_names)

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

#%%
import json
plant_path = 'D:/UQ Data Science/Subjects/Semester 4/DATA7902/Basil/New_Basil/leaf_and_plant/train_3_leaf_plant_crop_coco_instances.json' 
plant_img = 'D:/UQ Data Science/Subjects/Semester 4/DATA7902/Basil/New_Basil/leaf_and_plant/train_3'

# Load json from file
json_file = open(plant_path)
coco_json = json.load(json_file)
json_file.close()
#%%

from shapely.geometry import Polygon, MultiPolygon

instance_masks = []

img = cv2.imread(os.path.join(plant_img,'IMG_20210902_140626_2.jpg'))
segmentation = coco_json['annotations'][0]['segmentation']
segmentation = np.array(segmentation).reshape(-1,)
print(segmentation.shape)
w = img.shape[1]
h = img.shape[0]
mask = Image.new('1', (w,h))
mask_draw = ImageDraw.ImageDraw(mask, '1')

poly = mask_draw.polygon(segmentation, fill=1)
bool_array = np.array(mask) >0
instance_masks.append(bool_array)

#%%

img = train_img[0]
print(img.shape)
mask_seg = seg_img[0]
mask_seg = np.where(mask_seg>0,255,mask_seg)
print(mask_seg.shape)
bool_mask = mask_seg > 0
print(bool_mask.shape)