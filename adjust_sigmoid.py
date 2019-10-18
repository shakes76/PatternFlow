#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 13:59:11 2019

@author: khadekirti
"""

import tensorflow as tf 

def adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False):
    """Performs Sigmoid Correction on the input image.
    Also known as Contrast Adjustment.
    This function transforms the input image pixelwise according to the
    equation ``O = 1/(1 + exp*(gain*(cutoff - I)))`` after scaling each pixel
    to the range 0 to 1.
    Parameters
    ----------
    image : ndarray
        Input image.
    cutoff : float, optional
        Cutoff of the sigmoid function that shifts the characteristic curve
        in horizontal direction. Default value is 0.5.
    gain : float, optional
        The constant multiplier in exponential's power of sigmoid function.
        Default value is 10.
    inv : bool, optional
        If True, returns the negative sigmoid correction. Defaults to False.
    Returns
    -------
    out : ndarray
        Sigmoid corrected output image.
    See Also
    --------
    adjust_gamma
    References
    ----------
    .. [1] Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast
           Enhancement Functions",
           http://www.cis.rit.edu/fairchild/PDFs/PAP07.pdf
    """
    dtype_range = {'bool': (False, True),
                'bool_': (False, True),
                'float': (-1, 1),
                'float16': (-1, 1),
                'float32': (-1, 1),
                'float64': (-1, 1),
                'int16': (-32768, 32767),
                'int32': (-2147483648, 2147483647),
                'int64': (-9223372036854775808, 9223372036854775807),
                'int8': (-128, 127),
                'uint10': (0, 1023),
                'uint12': (0, 4095),
                'uint14': (0, 16383),
                'uint16': (0, 65535),
                'uint32': (0, 4294967295),
                'uint64': (0, 18446744073709551615),
                'uint8': (0, 255)}
    
    imin, imax = dtype_range[str(image.dtype)]
    
    if imax < 0: 
        imax =0 

    if imin < 0: 
        imin = 0          
        
    scale = float( imax  - imin ) 
    #Convert image 
    # Start a session 
    sess = tf.InteractiveSession()  
    # Convert array into tensor
    image = tf.convert_to_tensor(image)
    # Assert nonegative image 
    if tf.reduce_sum(tf.cast( tf.math.less( tf.cast(image, dtype = 'float64' ), tf.constant(0.0 , dtype = 'float64')), dtype = 'float64')).eval() != 0.0: 
        raise ValueError('Image Correction methods work correctly only on '
                         'images with non-negative values. Use '
                         'skimage.exposure.rescale_intensity.')      
 
    if inv: 
        out = (1 - 1 / (1 + tf.math.exp(gain * (cutoff - image / scale)))) * scale
    else: 
        out = (1 / (1 + tf.math.exp(gain * (cutoff - image / scale)))) * scale 
    
    return tf.cast(out ,image.dtype).eval()




  
