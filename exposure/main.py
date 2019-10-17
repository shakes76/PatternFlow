#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:52:35 2019

@author: Duc Phan
ID: 44040505
COMP3710
"""

import numpy as np
import tensorflow as tf
import exposureTF as etf
from skimage import exposure


import cv2

## Driver scrit
if __name__ == "__main__":
    image_ph = tf.placeholder(tf.uint8, shape = [None, None, 1])
    image_eq_hist = etf.tf_equalize_histogram(image_ph)
    
    #set the image to gray scale
    image = cv2.imread("dog1.jpg", 0)
    cv2.imwrite("original_grey.jpg", image)
    skimage_img = exposure.equalize_hist(image)
    cv2.imwrite("skimage_dog.jpg", skimage_img)
    image = np.reshape(image, (image.shape[0], image.shape[1], 1))
    with tf.Session() as sess:
       sess.run(tf.global_variables_initializer())
       tf_image = sess.run(image_eq_hist, feed_dict = {image_ph : image})
       
    print ("Ski dimension")
    print(exposure.equalize_hist(image).shape)
    
    print ("EXposure TF dimension")
    print(tf_image.shape)
    
    
    
    cv2.imshow("SKI ", skimage_img)
   
    
    cv2.imshow("TF", tf_image)
    cv2.imwrite("etf_dog.jpg", tf_image)
    
    cv2.waitKey()
    print("finish")