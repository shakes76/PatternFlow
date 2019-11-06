
"""
Author: Cameron Gordon, 42370057 
Licence: GNU General Public Licence v3.0
File provided free of copyright 
Date: 6 Nov 2019

Two algorithms from the Scikit Exposure module implemented in Tensorflow.

tf_intensity_range implements the intensity_range function
tf_rescale_intensity implements the rescale_intensity function
The algorithms are contained in the algorithm42370057.py file. 

A separate driver is contained within driver42370057.py. 
It calls an example image face and performs the tf_rescale_intenstiy function, plotting the output. """

import tensorflow as tf 

print(tf.version)


def tf_intensity_range(image,range_values='image',clip_negative=False): 
    """
    Author: Cameron Gordon, 42370057

    Return image intensity range (min, max) based on desired value type.
    Ported version of intensity_range in scikit exposure module.

    Parameters: 
    image: array 
    range_values: str or 2-tuple, optional (specified dtype, or intensity range)
    clip_negative: Bool, if True clip the image values to 0 

    """ 



    tf_dtype =tf.as_dtype(image.dtype) 
    tf_image = tf.convert_to_tensor(image,dtype=tf_dtype) # converts to tensor for tensorflow conversions 
    
    if range_values == 'dtype': 
        i_min, i_max = tf_dtype.limits 
        if clip_negative:
            i_min = 0 
        
    if range_values == 'image': 
        i_min = tf.reduce_min(tf_image)
        i_max = tf.reduce_max(tf_image) 

        

    elif type(range_values) is tuple: 

        i_min, i_max = range_values 
        
    return i_min, i_max 
    
def tf_rescale_intensity(image,in_range='image',out_range='dtype'): 
    
    """ 
    Author: Cameron Gordon, 42370057 

    Ported from rescale_intensity in scikit_exposure module.

    Return image after stretching or shrinking its intensity levels.
    The desired intensity range of the input and output, `in_range` and
    `out_range` respectively, are used to stretch or shrink the intensity range
    of the input image.

    Note: conversion to tf.float32 is required due to Tensorflow requirements 
    Note: Tensorflow 2.0 is required to be used for this module. 

    Parameters: 
    image: array 
    in_range, out_range : str or 2-tuple, optional
        Min and max intensity values of input and output image. 
        May be dtype or specified ranges. 
    """

    tf_dtype = tf.as_dtype(image.dtype) 
    tf_image = tf.convert_to_tensor(image,dtype=tf_dtype) # converts to tensor for tf conversions 
    
    imin, imax = tf_intensity_range(image,in_range) 

    omin, omax = tf_intensity_range(image,out_range,clip_negative=imin>0) # sets range for rescaling 

    imin = tf.cast(imin,tf.float32)
    imax = tf.cast(imax,tf.float32)
    omin = tf.cast(omin,tf.float32)
    omax = tf.cast(omax,tf.float32)
    tf_image = tf.cast(tf_image,tf.float32) # type cast required for output 

    image = tf.clip_by_value(tf_image,clip_value_min=imin,clip_value_max=imax)

    
    if imin != imax: 
        image = (image-imin)/(imax-imin) 
        
    return tf.cast(tf.constant(tf_image*(omax-omin)+omin,dtype=tf.float32),dtype=tf_dtype)

