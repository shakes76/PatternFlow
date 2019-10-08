# Tony Meng, Student No: 443298999
# ported from https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/transform/radon_transform.py#L12
# helper method https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/transform/_warps_cy.pyx

import tensorflow as tf

def _warp_fast(image, H):
    #output_shape = None
    #order = 1
    #mode = 'constant'
    #cval = 0
    
    #img = image
    #M = H
    #mode_c = 'C'

    rows = image.shape.as_list()[0]
    columns = image.shape.as_list()[1]
    out = tf.zeros([rows, columns], tf.float64)
    
    # bunch of function pointer code written in cython
    # assign functions to variables
    
    pass

def radon(image, theta = None, circle = True):
    if tf.rank(image) != 2:
        raise ValueError('The input image must be 2D')
    if theta is None:
        theta = list(range(180))
        #theta = tf.range(0, 180, 1)
    
    if circle:
        radius = min(image.shape.as_list()) // 2
        c = [list(range(image.shape.as_list()[0]))]
        c0 = tf.transpose(tf.constant(c))
        c1 = tf.constant(c)
        reconstruction_circle = ((c0 - image.shape.as_list()[0] // 2) ** 2
                                 + (c1 - image.shape.as_list()[1] // 2) ** 2)
        reconstruction_circle = reconstruction_circle <= radius ** 2
        pass
    else:
        pass
    
    #placeholder return
    return image