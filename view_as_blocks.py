#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:06:03 2019

@author: khadekirti
"""

import tensorflow as tf 
import numpy as np 


def view_as_blocks(arr_in, block_shape):
    # Check if input is a numpy or not 
    if not isinstance(arr_in,(np.ndarray)):
        raise TypeError('Input needs to be a numpy array')  
    # Check if the block is tuple 
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')  
    # start a session 
    sess = tf.InteractiveSession()  
    # Convert array into tensor
    arr_in = tf.convert_to_tensor(arr_in) 
    # check if all the block_shape elements is positive
    if not all(i >= 0 for i in block_shape): 
        raise ValueError("'block_shape' elements must be strictly positive")
    # Check if the shape of block matches that of array  
    if len(block_shape) != tf.shape(arr_in).shape:
        raise ValueError("'block_shape' must have the same length as 'arr_in.shape'") 
    # Check if the blockshape is compatible with the array 
    if sum([(arr_in.shape[i] % block_shape[i]) for i in range(len(block_shape))])  != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'") 
    # Get the new shape
    new_shape = [(arr_in.shape[i] // block_shape[i]) for i in range(len(block_shape))]
    new_shape.extend(list(block_shape))   
    # Get the shape of the array 0, 
    arr_out = tf.reshape( arr_in,new_shape)
    return arr_out.eval()
 
