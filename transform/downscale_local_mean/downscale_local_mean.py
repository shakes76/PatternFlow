#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 18:24:27 2019

@author: khadekirti
"""

import tensorflow as tf 
from view_as_blocks import view_as_blocks
 

def downscale_local_mean(image, factors, cval=0, clip=True):
    
    """Down-sample N-dimensional image by local averaging.
    The image is padded with `cval` if it is not perfectly divisible by the
    integer factors.
    In contrast to interpolation in `skimage.transform.resize` and
    `skimage.transform.rescale` this function calculates the local mean of
    elements in each block of size `factors` in the input image.
    Parameters
    ----------
    image : ndarray
        N-dimensional input image.
    factors : array_like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the
        integer factors.
    clip : bool, optional
        Unused, but kept here for API consistency with the other transforms
        in this module. (The local mean will never fall outside the range
        of values in the input image, assuming the provided `cval` also
        falls within that range.)
    Returns
    -------
    image : ndarray
        Down-sampled image with same number of dimensions as input image.
        For integer inputs, the output dtype will be ``float64``.
        See :func:`numpy.mean` for details.
    Examples
    --------
    >>> a = np.arange(15).reshape(3, 5)
    >>> a
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14]])
    >>> downscale_local_mean(a, (2, 3))
    array([[ 3.5,  4. ],
           [ 5.5,  4.5]])
    """ 
    

    # Check the instance of factors , if not through type error 
    if not isinstance(factors, tuple):
        raise TypeError('Factors needs to be a tuple')  
    
    # Start a session 
    sess = tf.InteractiveSession()  
    # Convert array into tensor
    image = tf.convert_to_tensor(image)
    
    # All factors needs to be greater then one, or else through value error 
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
 
