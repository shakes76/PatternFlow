#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""s
Created on Fri Oct 18 10:35:55 2019

@author: kajajuel
"""

from PIL import Image
import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt
from gaussian_two import make_gaussian_kernel_two, convolve_two

np_im = np.array(Image.open('gray_kitten.jpg'), dtype='float32')
tf_im = tf.constant(np_im.astype(np.float32))
img_4D = tf.reshape(tf_im, [tf_im.shape[0], tf_im.shape[1], tf_im.shape[2], 1])
print(tf_im.shape)
print(img_4D.shape)

gauss_kernel = make_gaussian_kernel_two(0.0, 0.2, 10)
kernel_4D = tf.reshape(gauss_kernel, [gauss_kernel.shape[0], gauss_kernel.shape[1], 1 ,1])
print(gauss_kernel.shape)
print(kernel_4D.shape)

c = convolve_two(img_4D, kernel_4D)
print(c.shape)