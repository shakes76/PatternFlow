#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""s
Created on Fri Oct 18 10:35:55 2019

@author: kajajuel
"""

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from gaussian import make_gaussian_kernel, convolve
import tensorflow as tf

tf.enable_eager_execution()


def reshape_to_fit_convolve(img, kernel):
    """
    Reshaping the image and the filter kernel to make it compatible with convolve(). 
    Arguments:
        img: an image object from the PIL Image module
        kernel: a 2D tensor
    """
    # saving image object as normalized numpy array
    np_img = np.array(img, dtype='float32') / 255. 
    
    tf_img = tf.constant(np_img.astype(np.float32))
    
    # reshaping
    img_4D = tf.reshape(tf_img, [tf_img.shape[0], tf_img.shape[1], tf_img.shape[2], 1])
    kernel_4D = tf.reshape(kernel, [kernel.shape[0], kernel.shape[1], 1 ,1])
    return img_4D, kernel_4D


imgage = Image.open('resources/gray_kitten.jpg')
plt.imshow(imgage)

gauss_kernel = make_gaussian_kernel(0.0, 0.1, 100)

img_reshaped, kernel_reshaped = reshape_to_fit_convolve(imgage, gauss_kernel)

c = convolve(img_reshaped, kernel_reshaped)
c_img = tf.reshape(c, [c.shape[0],c.shape[1], c.shape[2]])

plt.figure()
plt.imshow(c_img)