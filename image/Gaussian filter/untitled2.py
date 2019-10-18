#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 14:36:35 2019

@author: kajajuel
"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from gaussian import make_gaussian_kernel, convolve

img = Image.open('gray_kitten.jpg')
plt.imshow(img)

def reshape_para(img,kernel, rgb = True):
    """
    Reshape img to fit convolve
    """
    num_maps = 3
    
    if not rgb:
        print("not rgb")
        num_maps = 1       # set number of maps to 1
        img = img.convert('L', (0.2989, 0.5870, 0.1140, 0))  # convert to gray scale
    
    np_im = np.array(Image.open('gray_kitten.jpg'), dtype='float32') / 256.
    tf_im = tf.constant(np_im.astype(np.float32))
    
    kernel_4D = tf.reshape(kernel, [kernel.shape[0], kernel.shape[1], 1 ,1])
    img_4D = tf.reshape(tf_im, [tf_im.shape[0], tf_im.shape[1], tf_im.shape[2], num_maps])
    
    print("kernel_4D.shape: ", kernel_4D.shape)
    print("img_4D.shape: ", img_4D.shape)
    return kernel_4D, img_4D
    
                
gaussian = make_gaussian_kernel(0, 0.1, 10)
img, gaussian = reshape_para(img, gaussian, rgb = False)
conv = convolve(img, gaussian, rgb = False)

#conv_3D = tf.reshape(conv, [1200, 1600,3])
#
#print(conv_3D)
#plt.imshow(conv_3D)

#print(conv.shape)
#print(conv_3D.shape)
#
#plt.imshow(conv_3D)