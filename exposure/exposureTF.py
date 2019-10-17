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
    return: a tensor represeting the histogram of the image input
    """
    
    value_range = tf.constant([0., 255.], dtype = tf.float32)
    histogram = tf.histogram_fixed_width(tf.to_float(image), value_range, nbins)
    
    
    return histogram

    
def tf_cummulative_distribution(image, nbins = 256): 
    """
    ---------------
    input:  image - array
            nbins - optional, 256 by default
            the image is required to be gray scale
            
    ---------------
    return: a tensor that presents the cummulative distribution value
    
    """
    histogram = tf_histogram(image, nbins)
    cummulative_distribution = tf.cumsum(histogram)
    
    
    
    
    return cummulative_distribution[tf.reduce_min(tf.where(tf.greater(cummulative_distribution, 0)))]



def tf_equalize_histogram(image, nbins = 256): 
    """
    ---------------
    input:  image - array
            nbins - optional, 256 by default
            the image is grey scale 
            
    ---------------
    return: Float array
            Image array after histogram equalization.
    
    """
    
    hist = tf_histogram(image, nbins)
    cdf = tf.cumsum(hist)
    cdf_min = tf_cummulative_distribution(image, nbins)
    
    im_shape = tf.shape(image)
    pixel_count = im_shape[-3] * im_shape[-2]
    pixel_layout = tf.round(tf.to_float(cdf - cdf_min) * 255. /tf.to_float(pixel_count - 1))
    pixel_layout = tf.cast(pixel_layout, tf.uint8)
    
    
    
    return tf.expand_dims(tf.gather_nd(pixel_layout, tf.cast(image, tf.int32)), 2)


