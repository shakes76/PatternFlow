#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:06:03 2019

@author: khadekirti

Note: "as_stride" is a function taken from numpy, this is becuase: 

as_strided creates a view into the array given the exact strides and shape. 
This means it manipulates the internal data structure of ndarray and, 
if done incorrectly, the array elements can point to invalid 
memory and can corrupt results or crash your program. 
It is advisable to always use the original x.strides when calculating new strides
to avoid reliance on a contiguous memory layout.

Furthermore, arrays created with this function often contain self 
overlapping memory, so that two elements are identical. 
Vectorized write operations on such arrays will typically be unpredictable. 
They may even give different results for small, large, or transposed arrays. 
Since writing to these arrays has to be tested and done with great care,
you may want to use writeable=False to avoid accidental write operations.
For these reasons it is advisable to avoid as_strided when possible.     

Reference - https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html   
"""

import tensorflow as tf 
import numpy as np 
from numpy.lib.stride_tricks import as_strided
 


def view_as_blocks(arr_in, block_shape):
    """Block view of the input n-dimensional array (using re-striding).
    
    Blocks are non-overlapping views of the input array.
    
    Parameters
    ----------
    arr_in : ndarray
        N-d input array.
    block_shape : tuple
        The shape of the block. Each dimension must divide evenly into the
        corresponding dimensions of `arr_in`.
    
    Returns
    -------
    arr_out : ndarray
        Block view of the input array.
    
    Examples
    --------
    >>> import numpy as np
    >>> from skimage.util.shape import view_as_blocks
    >>> A = np.arange(4*4).reshape(4,4)
    >>> A
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11],
           [12, 13, 14, 15]])
    >>> B = view_as_blocks(A, block_shape=(2, 2))
    >>> B[0, 0]
    array([[0, 1],
           [4, 5]])
    >>> B[0, 1]
    array([[2, 3],
           [6, 7]])
    >>> B[1, 0, 1, 1]
    13
    >>> A = np.arange(4*4*6).reshape(4,4,6)
    >>> A  # doctest: +NORMALIZE_WHITESPACE
    array([[[ 0,  1,  2,  3,  4,  5],
            [ 6,  7,  8,  9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23]],
           [[24, 25, 26, 27, 28, 29],
            [30, 31, 32, 33, 34, 35],
            [36, 37, 38, 39, 40, 41],
            [42, 43, 44, 45, 46, 47]],
           [[48, 49, 50, 51, 52, 53],
            [54, 55, 56, 57, 58, 59],
            [60, 61, 62, 63, 64, 65],
            [66, 67, 68, 69, 70, 71]],
           [[72, 73, 74, 75, 76, 77],
            [78, 79, 80, 81, 82, 83],
            [84, 85, 86, 87, 88, 89],
            [90, 91, 92, 93, 94, 95]]])
    >>> B = view_as_blocks(A, block_shape=(1, 2, 2))
    >>> B.shape
    (4, 2, 3, 1, 2, 2)
    >>> B[2:, 0, 2]  # doctest: +NORMALIZE_WHITESPACE
    array([[[[52, 53],
             [58, 59]]],
           [[[76, 77],
             [82, 83]]]])
    """
    
    # Check if input is a numpy, if not through type error.  
    if not isinstance(arr_in,(np.ndarray)):
        raise TypeError('Input needs to be a numpy array')  
    # Check if the block is tuple, if not, through type error 
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple') 
    # Check if all the block_shape elements is positive
    if not all(i >= 0 for i in block_shape): 
        raise ValueError("'block_shape' elements must be strictly positive")     
    
    # Start a tensorflow session 
    sess = tf.InteractiveSession()  
    # Convert array into tensor
    arr_in = tf.convert_to_tensor(arr_in) 
    
    # Check if the shape of block matches that of array,if not through value error   
    if len(block_shape) != tf.shape(arr_in).shape:
        raise ValueError("'block_shape' must have the same length as 'arr_in.shape'") 
    # Check if the blockshape is compatible with the array,if not through value error 
    if sum([(arr_in.shape[i] % block_shape[i]) for i in range(len(block_shape))]) != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'") 
    
    # Get the new shape
    new_shape = [(arr_in.shape[i] // block_shape[i]) for i in range(len(block_shape))]
    new_shape.extend(list(block_shape))   
    # Get the shape of the array using numpy here
    new_strides = tuple(arr_in.eval().strides * np.array(block_shape))+ arr_in.eval().strides
    arr_out = as_strided(arr_in.eval(), shape=new_shape, strides=new_strides) 
    return arr_out
 


