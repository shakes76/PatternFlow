# Tony Meng, Student No: 443298999
# ported from https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/transform/radon_transform.py#L12
# helper method https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/transform/_warps_cy.pyx
# more from https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/_shared/interpolation.pxd

import tensorflow as tf
import math

def get_pixel2d(image, rows, cols, r, c, cval):
    """
    Get a pixel from the image, using Constant wrapping mode.
    
    Parameters
    ----------
    image :
        Input image.
    rows, cols :
        Shape of image.
    r, c :
        Position at which to get the pixel.
    cval :
        Constant value to use for constant mode.
    
    Returns
    -------
    value :
        Pixel value at given position.
    """
    # mode = 'C' (constant)
    if (r < 0) or (r >= rows) or (c < 0) or (c >= cols):
        return cval
    else:
        return image[r][c] # will return a scalar tensor

def bilinear_interpolation(image, rows, cols, r, c, cval):
    """
    Bilinear interpolation at a given position in the image.
    
    Parameters
    ----------
    image :
        Input image.
    rows, cols :
        Shape of image.
    r, c :
        Position at which to interpolate.
    cval : numeric
        Constant value to use for constant mode.
    
    Returns
    -------
    value :
        Interpolated value.
    """
    # mode = 'C' (constant)
    minr = math.floor(r)
    minc = math.floor(c)
    maxr = math.ceil(r)
    maxc = math.ceil(c)
    dr = r - minr
    dc = c - minc
    
    top_left = get_pixel2d(image, rows, cols, minr, minc, cval)
    top_right = get_pixel2d(image, rows, cols, minr, maxc, cval)
    bottom_left = get_pixel2d(image, rows, cols, maxr, minc, cval)
    bottom_right = get_pixel2d(image, rows, cols, maxr, maxc, cval)
    
    top = (1 - dc) * top_left + dc * top_right
    bottom = (1 - dc) * bottom_left + dc * bottom_right
    return ((1 - dr) * top + dr * bottom)


def _transform_metric(x, y, H, x_, y_):
    pass

def _transform_affine(x, y, H, x_, y_):
    pass

def _transform_projective(x, y, H, x_, y_):
    pass

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
    # tf.rank does not return the correct value if eager execution is off
    if len(image.shape.as_list()) != 2:
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