#!/usr/bin/env python
# coding: utf-8

# In[15]:


# load image
isic_input = glob.glob("ISIC-2017_Training_Data/*.jpg")
isic_groundtruth = glob.glob("ISIC-2017_Training_Part1_GroundTruth/*.png")


# In[17]:


len(isic_groundtruth)


# In[3]:


from keras.utils import normalize
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[18]:


transformed_X = 256
transformed_Y = 256
def load_images(path):
    image_list = []
    for fi in os.listdir(path):
    #print(fi)
        if fi.endswith(".csv") or "superpixels" in fi:
            continue
        img = cv2.imread(os.path.join(path, fi),cv2.IMREAD_COLOR)
        img = cv2.resize(img,(transformed_Y,transformed_X))
        img = img / 255.0
        img = img.astype(np.float32)
        image_list.append(img)
    image_list = np.array(image_list)
    return image_list


# In[19]:


def load_masks(path):
    masks_list = []
    for fi in os.listdir(path):
    #print(fi)
        if fi.endswith(".csv") or "superpixels" in fi:
            continue
        mask = cv2.imread(os.path.join(path, fi),cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask,(transformed_Y,transformed_X),interpolation = cv2.INTER_NEAREST)
        mask = mask / 255.0
        mask = mask.astype(np.float32)
        masks_list.append(mask)
    masks_list = np.array(masks_list)
    return masks_list


# In[ ]:


X_train = load_images("ISIC-2017_Training_Data/")
masks_train_images = load_masks("ISIC-2017_Training_Part1_GroundTruth/")


# In[ ]:


x_validate = load_images("ISIC-2017_Validation_Data/")
masks_valid_images = load_masks("ISIC-2017_Validation_Part1_GroundTruth")


# In[ ]:


x_test = load_images("ISIC-2017_Test_v2_Data/")
masks_test_images = load_images("ISIC-2017_Test_v2_Part1_GroundTruth")

