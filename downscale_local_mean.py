#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:24:27 2019

@author: khadekirti
"""

import tensorflow as tf 
import numpy as np
from view_as_blocks import view_as_blocks
 

def downscale_local_mean(image, factors, cval=0, clip=True):
    # Check if input is a numpy or not, if not through type error
    if not isinstance(image,(np.ndarray)):
        raise TypeError('Input needs to be a numpy array')   
    # Check the instance of factors , if not through type error 
    if not isinstance(factors, tuple):
        raise TypeError('Factors needs to be a tuple')  
    
    # Start a session 
    sess = tf.InteractiveSession()  
    # Convert array into tensor
    image = tf.convert_to_tensor(image)
    
    # All factors needs to be greater then zero, or else through value error 
    if not all(i >= 1 for i in factors): 
        raise ValueError("factors elements must be strictly positive and greater than 1") 
    # Check the shape of block_size and image, or else through value error 
    if len(factors) != tf.shape(image).shape:
        raise ValueError("`factor must have the same length as `image.shape`.")  
    # Define Padding width    
    pad_width = [] 
    for i in range(len(factors)): 
        if (image.shape[i] % abs(factors[i])) != 0:
            after_width = abs(factors[i]) - (image.shape[i] % abs(factors[i])) 
        else:
            after_width = tf.constant(0)
        pad_width.append((0, after_width))    
    # Pad the input  
    image = tf.pad(image,paddings =pad_width ,mode='CONSTANT',constant_values=cval)
    # Get the view of the input 
    blocked = view_as_blocks(image.eval(), factors)  
    # Reduce the input 
    re = tf.math.reduce_mean( tf.dtypes.cast(blocked, 'float64',) , axis=tuple(range(tf.shape(image).shape[0], tf.shape(blocked).shape[0])))
    return re.eval()
 
