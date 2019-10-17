#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:13:49 2019

@author: ethanphan
"""

import tensorflow as tf
import numpy as np #only use to read 




def tf_histogram(image, nbins = 256):

    """
    # Read the image file, this has approved by the the course coordinator
    np_im = np.array(Image.open(image_file), dtype='float32')
    tf_im = tf.constant(np_im.astype(np.float32))
    """
    value_range = tf.constant([0., 255.], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(tf.to_float(image), value_range, nbins)
    
    #For this method we always flatten the image  
    
    """
    sh = image.shape
    if len(sh == 3) and sh[-1] < 4: 
        print("his might be a color image. The histogram will be "
             "computed on the flattened image. You can instead "
             "apply this function to each color channel.")
    """
    
    return 

