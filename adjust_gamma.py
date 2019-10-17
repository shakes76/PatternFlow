#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 00:51:17 2019

@author: khadekirti
"""

import tensorflow as tf 

def adjust_gamma(image, gamma=1, gain=1):
    """Performs Gamma Correction on the input image.
    Also known as Power Law Transform.
    This function transforms the input image pixelwise according to the
    equation ``O = I**gamma`` after scaling each pixel to the range 0 to 1.
    Parameters
    ----------
    image : ndarray
        Input image.
    gamma : float, optional
        Non negative real number. Default value is 1.
    gain : float, optional
        The constant multiplier. Default value is 1.
    Returns
    -------
    out : ndarray
        Gamma corrected output image.
    See Also
    --------
    adjust_log
    Notes
    -----
    For gamma greater than 1, the histogram will shift towards left and
    the output image will be darker than the input image.
    For gamma less than 1, the histogram will shift towards right and
    the output image will be brighter than the input image.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gamma_correction
    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.moon())
    >>> gamma_corrected = exposure.adjust_gamma(image, 2)
    >>> # Output is darker for gamma > 1
    >>> image.mean() > gamma_corrected.mean()
    True
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
    
    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number.")
        
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
    out = ((image / scale) ** gamma) * scale * gain
    return out.eval()






