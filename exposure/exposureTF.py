#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 15:13:49 2019

@author: ethanphan
"""

import tensorflow as tf


def tf_histogram(image, nbins = 256):
    """
    -----------------
    input: image array
            nbins: optional, 256 by default
            
    ------------------
    return: a tensor
    """
    
    
    
    """
    # Read the image file, this has approved by the the course coordinator
    np_im = np.array(Image.open(image_file), dtype='float32')
    tf_im = tf.constant(np_im.astype(np.float32))
    """
    value_range = tf.constant([0., 255.], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(tf.to_float(image), value_range, nbins)
    
    #For this method we always flatten the image  
    
    
    return histogram

    
def tf_cummulative_distribution(image, nbins = 256): 
    """
    ---------------
    input:  image - array
            nbins - optional, 256 by default
            
            
    ---------------
    return: tensor
    
    """
    histogram = tf_histogram(image, nbins)
    cummulative_distribution = tf.cumsum(histogram)
    
    
    
    
    return cummulative_distribution[tf.reduce_min(tf.where(tf.greater(cummulative_distribution, 0)))]




