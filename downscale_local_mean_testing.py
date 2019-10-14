#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:24:54 2019

@author: khadekirti
"""

import os 
os.chdir('/Users/khadekirti/Desktop/Pattern Recognistion/Project')
  

 
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

def downscale_local_mean(image, factors, cval=0, clip=True):
    # Convert to tensorflow     
    image = tf.convert_to_tensor(image)
    block_size = tf.convert_to_tensor(factors) 
    # Check the shape of block_size and image 
    if block_size.shape != tf.shape(image).shape:
        raise ValueError("`block_size` must have the same length "
                         "as `image.shape`.")  
    # Padding width    
    pad_width = []
    for i in range(tf.shape(block_size).eval()[0]):     
        # check if all the downsampling factor is greater than 1 
        if not tf.math.reduce_all(tf.math.greater(block_size[i], 1)).eval() : 
            raise ValueError("Down-sampling factors must be >= 1. Use `skimage.transform.resize` to up-sample an image.")
        if tf.mod(tf.shape(image[i]),block_size[i]) != 0:
            after_width = tf.math.subtract( block_size[i], tf.mod(tf.shape(image[i]),block_size[i])[0]) 
        else:
            after_width = tf.constant(0)
        pad_width.append((0, after_width.eval()))    
    image = tf.pad(image,paddings = tf.constant(pad_width),mode='CONSTANT',constant_values=cval)
    blocked = view_as_blocks(image.eval(), tuple(block_size.eval())) 
    re = tf.math.reduce_mean(tf.convert_to_tensor(blocked, dtype = tf.float64), axis=tuple(range(image.eval().ndim, blocked.ndim))).eval()
    return re  



sess = tf.InteractiveSession()  
 

a = np.arange(15).reshape(3, 5)
image = a 
block_size = (2,3) 

b = downscale_local_mean(image, block_size,cval = 0)
