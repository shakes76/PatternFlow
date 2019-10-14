#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:06:03 2019

@author: khadekirti
"""

import tensorflow as tf 
 
def view_as_blocks(arr_in, block_shape):
    if not tf.is_tensor(block_shape) and tf.is_tensor(arr_in):
        raise TypeError('Please input tensorflow') 
    # check if all the block_shape elements is positive
    if not sess.run(tf.math.reduce_all(tf.math.greater( block_shape, 0))) : 
        raise ValueError("'block_shape' elements must be strictly positive")
    # Check if the shape of block matches that of array  
    if block_shape.shape != tf.shape(arr_in).shape:
        raise ValueError("'block_shape' must have the same length as 'arr_in.shape'") 
    # Check if the blockshape is compatible with the array 
    if sess.run(tf.math.reduce_sum(tf.mod(tf.shape(arr_in),block_shape))) != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'") 
    # Get the new shape
    new_shape = tf.concat([tf.math.floordiv(tf.shape(arr_in),block_shape), block_shape], 0)   
    # Get the shape of the array 0
    arr_out = tf.reshape( arr_in,new_shape)
    return arr_out 
