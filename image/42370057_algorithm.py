try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass

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
    tf_image = tf.convert_to_tensor(image,dtype=tf_dtype) 
    
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
    tf_image = tf.convert_to_tensor(image,dtype=tf_dtype) 
    
    imin, imax = tf_intensity_range(image,in_range) 

    omin, omax = tf_intensity_range(image,out_range,clip_negative=imin>0)

    imin = tf.cast(imin,tf.float32)
    imax = tf.cast(imax,tf.float32)
    omin = tf.cast(omin,tf.float32)
    omax = tf.cast(omax,tf.float32)
    tf_image = tf.cast(tf_image,tf.float32)

    image = tf.clip_by_value(tf_image,clip_value_min=imin,clip_value_max=imax)

    
    if imin != imax: 
        image = (image-imin)/(imax-imin) 
        
    return tf.cast(tf.constant(tf_image*(omax-omin)+omin,dtype=tf.float32),dtype=tf_dtype)


import matplotlib.pyplot as plt 
from scipy import misc
# uses the default scipy raccoon face as the example image 

def main(): 
    """
    Author: Cameron Gordon, 42370057
    Performs a test using the scipy raccoon face as an example image.
    Then calls intensity_test, which is then plotted. 
    """ 
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(1,2,1)
    face = misc.face()
    plt.imshow(face)
    
    image = face # example image 
    intensity_test = tf_rescale_intensity(image) 
    ax = plt.subplot(1,2,2)
    plt.imshow(intensity_test)
    plt.show()

main()
