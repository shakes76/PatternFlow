# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from tqdm import tqdm_notebook
from skimage.transform import resize

# Loading lesion images for ISIC data set
def loading_img(img_path,img_width,img_height):
    ids_all = next(os.walk(img_path))[2]
    ids_all_sort = sorted(ids_all)
    X = np.zeros((len(ids_all_sort), img_height, img_width, 1), dtype=np.float32)
    for n, id_ in tqdm_notebook(enumerate(ids_all_sort), total=len(ids_all_sort)):
        # Load images
        img = load_img(img_path+id_, color_mode = 'grayscale')
        x_img = img_to_array(img)
        x_img = resize(x_img, (img_height, img_width, 1), mode = 'constant', preserve_range = True)
        X[n] = x_img/255
        
    return X

# Loading segmentation images for ISIC data set
def loading_seg(seg_path,img_width,img_height):
    ids_all = next(os.walk(seg_path))[2]
    ids_all_sort = sorted(ids_all)
    X = np.zeros((len(ids_all_sort), img_height, img_width, 1), dtype=np.float32)
    for n, id_ in tqdm_notebook(enumerate(ids_all_sort), total=len(ids_all_sort)):
        # Load images
        img = load_img(seg_path+id_, color_mode = 'grayscale')
        x_img = img_to_array(img)
        x_img = resize(x_img, (img_height, img_width, 1), mode = 'constant', preserve_range = True)
        X[n] = x_img
        
    return X