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

#img = Image.open('gray_kitten.jpg')
#plt.imshow(img)

np_im = np.array(Image.open('gray_kitten.jpg'), dtype='float32')
tf_im = tf.constant(np_im.astype(np.float32))

gaussian = make_gaussian_kernel(0, 0.1, 10)
conv = convolve(tf_im, gaussian)

conv_3D = tf.reshape(conv, [1200, 1600,3])

print(tf_im.shape)
print(conv.shape)
print(conv_3D.shape)
