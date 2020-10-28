# -*- coding: utf-8 -*-
"""
UNet Model for segmenting ISIC images.

@author: s4537175
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import os
import random
import math

#%%

### VARIABLES TO CHANGE ###

# Image folder location
root_folder = 'C:\\Users\\s4537175\\Downloads\\COMP3710'

# Proportions of image set to use
total_prop = 1
val_prop = 0.1
test_prop = 0.1

#%%

### Load and shuffle image filenames ###

input_path = root_folder + os.path.sep + 'ISIC2018_Task1-2_Training_Input_x2' 
ground_truth_path = root_folder + os.path.sep + 'ISIC2018_Task1_Training_GroundTruth_x2' 
#output_path = root_folder + os.path.sep + 'UNet_results'

imgs = glob.glob(input_path + os.path.sep + '*.jpg')
segs = glob.glob(ground_truth_path + os.path.sep + '*.png')

seed = random.random()
random.seed(seed)
random.shuffle(imgs)
random.seed(seed)
random.shuffle(segs)

#print(imgs[0:5])
#print(segs[0:5])
#print(len(imgs))

#%%

# Reduce total number of images to total_prop of original 

new_img_num = math.floor(len(imgs) * total_prop)

imgs = imgs[0 : new_img_num]
segs = segs[0 : new_img_num]

print('Total image set size: ' + str(len(imgs)))

# Split remaining images into traing, validation and test sets, according to
# val_prop and test_prop

train_prop = 1 - val_prop - test_prop
train_num = math.ceil(len(imgs) * train_prop)

train_img = imgs[0 : train_num]
train_seg = segs[0 : train_num]

val_num = math.floor(len(imgs) * val_prop)

val_img = imgs[train_num : train_num + val_num]
val_seg = segs[train_num : train_num + val_num]

test_img = imgs[train_num + val_num : len(imgs)]
test_seg = segs[train_num + val_num : len(imgs)]

print('Training set size: ' + str(len(train_img)))
print('Validation set size: ' + str(len(val_img)))
print('Test set size: ' + str(len(test_img)))

#%%

# Make datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_img, train_seg))
val_ds = tf.data.Dataset.from_tensor_slices((val_img, val_seg))
test_ds = tf.data.Dataset.from_tensor_slices((test_img, test_seg))

# Shuffle datasets
train_ds = train_ds.shuffle(len(train_img))
val_ds = val_ds.shuffle(len(val_img))


#%%
'''
def find_min_size(files):
    min_h = 2000
    min_w = 2000
    portrait = 0;
    for i in range(len(files)):
        print(i)
        img = tf.io.read_file(files[i])
        img = tf.image.decode_png(img, channels=3)
        h = img.shape[0]
        w = img.shape[1]
        if w < h :
            portrait += 1
            temp = w
            w = h
            h = temp
        if h < min_h :
            min_h = img.shape[0]
        if w < min_w :
            min_w = img.shape[1]
    print(min_h)
    print(min_w)
    print(portrait)

find_min_size(imgs)
'''

#%%


def load_single(filename, c):
    img = tf.io.read_file(filename)
    img = tf.image.decode_png(img, channels=c)
    print(img.shape)
    if (img.shape[0] < img.shape[1]) :
        tf.image.transpose(img)
    img = tf.image.resize(img, (270, 288))
    img = tf.cast(img, tf.float32)
    print(img.shape)
    return img / 255
    
x = load_single(train_img[0], 3)
plt.imshow(x)

#%%

x = load_single(train_seg[0], 1)
y = (x > 0.5)
print(y)
plt.imshow(y, cmap='gray')














