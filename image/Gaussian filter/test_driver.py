#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""s
Created on Fri Oct 18 10:35:55 2019

@author: kajajuel
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from gaussian import convolve
import tensorflow as tf

tf.enable_eager_execution()


def reshape_image(img):
    """
    Reshaping the image to make it compatible with convolve(). 
    Arguments:
        img: an image object from the PIL Image module
    """
    # saving image object as normalized numpy array
    np_img = np.array(img, dtype='float32') / 255. 
    
    tf_img = tf.constant(np_img.astype(np.float32))
    
    # reshaping
    #img_4D = tf.reshape(tf_img, [1, tf_img.shape[0], tf_img.shape[1], 3])
    img_4D = tf.reshape(tf_img, [tf_img.shape[0], tf_img.shape[1], tf_img.shape[2], 1])
    
    return img_4D


imgage = Image.open('resources/gray_kitten.jpg')
plt.imshow(imgage)
img_reshaped = reshape_image(imgage)

c = convolve(img_reshaped, 0.5, 0.3, 10)

c_img = tf.reshape(c, [c.shape[0],c.shape[1], c.shape[2]])

plt.figure()
plt.imshow(c_img)