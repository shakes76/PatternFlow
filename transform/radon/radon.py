# Tony Meng, Student No: 443298999
# ported from https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/transform/radon_transform.py#L12
# helper method https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/transform/_warps_cy.pyx
# more from https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/_shared/interpolation.pxd

import tensorflow as tf
import math

def matrix_multiply(X, Y):
    result = [[0,0,0],[0,0,0],[0,0,0]]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                result[i][j] += X[i][k] * Y[k][j]
    return result

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
    
    return image

def radon(image, theta = None, circle = True):
    # tf.rank does not return the correct value if eager execution is off
    imageShape = image.shape.as_list()
    if len(imageShape) != 2:
        raise ValueError('The input image must be 2D')
    if theta is None:
        theta = list(range(180))
        #theta = tf.range(0, 180, 1)
    
    if circle:
        radius = min(imageShape) // 2
        c = [list(range(imageShape[0]))]
        c0 = tf.transpose(tf.constant(c))
        c1 = tf.constant(c)
        reconstruction_circle = ((c0 - imageShape[0] // 2) ** 2
                                 + (c1 - imageShape[1] // 2) ** 2)
        reconstruction_circle = reconstruction_circle <= radius ** 2
        slices = []
        for d in (0, 1):
            if imageShape[d] > min(imageShape):
                excess = imageShape[d] - min(imageShape)
                slices.append(slice(math.ceil(excess / 2),
                                    math.ceil(excess / 2) + min(imageShape)))
            else:
                slices.append(slice(None))
        slices = tuple(slices)
        padded_image = image[slices]
    else:
        # TODO
        pass
    
    #print(padded_image.shape)
    radon_image = tf.zeros([padded_image.shape.as_list()[0], len(theta)])
    center = padded_image.shape.as_list()[0] // 2
    
    shift0 = [[1, 0, -center],
              [0, 1, -center],
              [0, 0, 1]]
    shift1 = [[1, 0, center],
              [0, 1, center],
              [0, 0, 1]]
    
    def build_rotation(theta):
        T = math.radians(theta)
        R = [[math.cos(T), math.sin(T), 0],
             [-math.sin(T), math.cos(T), 0],
             [0, 0, 1]]
        # return shift1 * R * shift0
        return matrix_multiply(matrix_multiply(shift1, R), shift0)
    
    radon_image_cols = []
    for i in range(len(theta)):
        rotated = _warp_fast(padded_image, build_rotation(theta[i]))
        col = tf.reduce_sum(rotated, 0)
        col = tf.expand_dims(col, 1)
        radon_image_cols.append(col)
        
    radon_image = tf.concat(radon_image_cols, 1)
    return radon_image