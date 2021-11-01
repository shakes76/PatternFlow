# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:01:46 2021

@author: siwan
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

img_path = './ISIC_Images'
mask_path = './ISIC_Masks'
anno_file = './anno.txt'
img_width = 511
img_height = 384
img_num = 16073

images = tf.keras.utils.image_dataset_from_directory(img_path,image_size=(img_height, img_width))
masks = tf.keras.utils.image_dataset_from_directory(mask_path,image_size=(img_height, img_width))

def correct_yolo_box(img_num):
    filepath = f"{mask_path}/ISIC2018_Task1_Training_GroundTruth_x2/ISIC_{img_num:07d}_segmentation.png"
    mask = load_img(filepath, color_mode = "grayscale")
    mask = img_to_array(mask)
    mask /= 255
    mask = np.squeeze(mask)
    img_height, img_width = mask.shape
    min_x = None
    max_x = None
    max_y = None
    min_y = None
    for i in range(0, img_height):
        if 1 in mask[i]:
            min_y = i
            break
    for i in range(img_height - 1, -1, -1):
        if 1 in mask[i]:
            max_y = i
            break
    for i in range(0, img_width):
        if 1 in mask.T[i]:
            min_x = i
            break
    for i in range(img_width - 1, -1, -1):
        if 1 in mask.T[i]:
            max_x = i
            break
    if min_y == None or max_y == None or min_x == None or max_x == None:
        print("Image is not a valid mask")
        raise
    return [min_x, min_y, max_x, max_y]

list_file= open(anno_file, 'w')
for i in range(img_num):
    try:
        coords = correct_yolo_box(i)
        file_name = f"ISIC_{img_num:07d}.jpg"
    except:
        continue
    list_file.write(file_name + " " + ",".join([str(a) for a in b]) + ",0\n")
list_file.close()