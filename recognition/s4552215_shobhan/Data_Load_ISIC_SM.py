# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:48:56 2020

@author: s4552215
"""


import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize

# To load lesion images
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

# To load segmentation images from ISICs dataset
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
