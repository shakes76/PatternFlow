#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:24:27 2019

@author: khadekirti
"""

import tensorflow as tf 
from view_as_blocks import view_as_blocks
 

def downscale_local_mean(image, factors, cval=0, clip=True):
    # check the instance of factors 
    if not isinstance(factors, tuple):
        raise TypeError('factors needs to be a tuple')  
    # Check if input is tensor 
    if not tf.is_tensor(image):
        raise TypeError('input needs to be a Tensorflow')  
    # All block
    if not all(i >= 1 for i in factors): 
        raise ValueError("factors elements must be strictly positive and greater than 1") 
        
    # Check the shape of block_size and image 
    if len(factors) != tf.shape(image).shape:
        raise ValueError("`block_size` must have the same length "
                         "as `image.shape`.")  
    # Padding width    
    pad_width = [] 
    for i in range(len(factors)): 
        if (image.shape[i] % abs(factors[i])) != 0:
            after_width = abs(factors[i]) - (image.shape[i] % abs(factors[i])) 
        else:
            after_width = tf.constant(0)
        pad_width.append((0, after_width))    
    image = tf.pad(image,paddings =pad_width ,mode='CONSTANT',constant_values=cval)
    blocked = view_as_blocks(image, factors)  
    re = tf.math.reduce_mean( tf.dtypes.cast(blocked, 'float64',) , axis=tuple(range(tf.shape(image).shape[0], tf.shape(blocked).shape[0])))
    return re
 
