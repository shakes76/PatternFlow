# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 2020

@author: Md Abdul Bari
"""

import os
import numpy as np
import tqdm.notebook 
from skimage.transform import resize
from keras.preprocessing.image import img_to_array, load_img


def load_image(image_path=None):
    """Resizes images and stores pixel values in a multidimensional array
    
    Parameter: 
        directory (str): path for the image source
        
    Return:
        a multi-dimensional array[[float]]: having pixel values of the images
    """
    if image_path == None:
        image_path = input()
    # resize image dimension
    IMG_HEIGHT = IMG_WIDTH = 256
    ids = next(os.walk(image_path))[2] 
    ids = sorted(ids)
    X = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
    for n, id_ in tqdm.notebook.tqdm(enumerate(ids), total=len(ids)):
        # load input images
        img = load_img(image_path + "/" + id_, color_mode = "grayscale")
        x_img = img_to_array(img)
        x_img = resize(x_img, (IMG_HEIGHT, IMG_WIDTH, 1), mode = 'constant', 
                       preserve_range = True)
        # store images
        X[n] = x_img
    return X