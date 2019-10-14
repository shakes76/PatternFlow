#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:08:45 2019

@author: khadekirti
"""

import os 
os.chdir('/Users/khadekirti/Desktop/Pattern Recognistion/Project')
 
import tensorflow as tf
import numpy as np 
sess = tf.InteractiveSession()  

import view_as_blocks 


def view_as_blocks(arr_in, block_shape):
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple') 
    # Convert to tensorflow     
    arr_in = tf.convert_to_tensor(arr_in)
    block_shape = tf.convert_to_tensor(block_shape)
    # check if all the block_shape elements is positive
    if not tf.math.reduce_all(tf.math.greater( block_shape, 0)).eval() : 
        raise ValueError("'block_shape' elements must be strictly positive")
    # Check if the shape of block matches that of array  
    if block_shape.shape != tf.shape(arr_in).shape:
        raise ValueError("'block_shape' must have the same length as 'arr_in.shape'") 
    # Check if the blockshape is compatible with the array 
    if tf.math.reduce_sum(tf.mod(tf.shape(arr_in),block_shape)).eval() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'") 
    # Get the new shape
    new_shape = tf.concat([tf.math.floordiv(tf.shape(arr_in),block_shape), block_shape], 0)   
    # Get the shape of the array 0
    arr_out = tf.reshape( arr_in,new_shape.eval())
    return arr_out.eval()  


A = np.arange(4*4*6).reshape(4,4,6)  
B = view_as_blocks(A,(1, 2, 2)) 
