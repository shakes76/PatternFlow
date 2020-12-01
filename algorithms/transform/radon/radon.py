"""
Tony Meng, Student No: 443298999
Ported to tensorflow from the scikit-image implementation using numpy.

https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/transform/radon_transform.py#L12
https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/transform/_warps_cy.pyx
https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/_shared/interpolation.pxd
"""

import tensorflow as tf
import math

def matrix_multiply(X, Y):
    """
    Does matrix multiplication on two 3x3 integer arrays
    
    Parameters
    ----------
    X, Y :
        3x3 integer arrays
    
    Returns
    -------
    result :
        3x3 integer array equal to XY
    """
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
        return tf.constant(cval, tf.float64)
    else:
        return tf.cast(image[r][c], tf.float64) # will return a scalar tensor

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

def _transform_metric(x, y, H):
    """
    Apply a metric transformation to a coordinate.
    
    Parameters
    ----------
    x, y :
        Input coordinate.
    H :
        3x3 transformation matrix.
    
    Returns
    -------
    (x_, y_) :
        Tuple of output coordinates
    """
    x_ = H[0][0] * x + H[0][2]
    y_ = H[1][1] * y + H[1][2]
    return (x_, y_)

def _transform_affine(x, y, H):
    """
    Apply an affine transformation to a coordinate.
    
    Parameters
    ----------
    x, y :
        Input coordinate.
    H :
        3x3 transformation matrix.
    
    Returns
    -------
    (x_, y_) :
        Tuple of output coordinates
    """
    x_ = (H[0][0] * x) + (H[0][1] * y) + H[0][2]
    y_ = (H[1][0] * x) + (H[1][1] * y) + H[1][2]
    return (x_, y_)

def _transform_projective(x, y, H):
    """
    Apply a homography to a coordinate.
    
    Parameters
    ----------
    x, y :
        Input coordinate.
    H :
        3x3 transformation matrix.
    
    Returns
    -------
    (x_, y_) :
        Tuple of output coordinates
    """
    z_ = H[2][0] * x + H[2][1] * y + H[2][2]
    x_ = (H[0][0] * x + H[0][1] * y + H[0][2]) / z_
    y_ = (H[1][0] * x + H[1][1] * y + H[1][2]) / z_
    return (x_, y_)

def _warp_fast(image, H):
    """
    Projective transformation (homography).
    
    Perform a projective transformation (homography) of a
    floating point image, using interpolation.
    
    For each pixel, given its homogeneous coordinate :math:`\mathbf{x}
    = [x, y, 1]^T`, its target position is calculated by multiplying
    with the given matrix, :math:`H`, to give :math:`H \mathbf{x}`.
    E.g., to rotate by theta degrees clockwise, the matrix should be::
    
      [[cos(theta) -sin(theta) 0]
       [sin(theta)  cos(theta) 0]
       [0            0         1]]
    
    or, to translate x by 10 and y by 20::
    
      [[1 0 10]
       [0 1 20]
       [0 0 1 ]].
    
    Parameters
    ----------
    image : 2-D tensor
        Input image.
    H : 3x3 array
        transformation matrix H that defines the homography.
    
    Notes
    -----
    output_shape = None
    order = 1
    mode = 'constant'
    cval = 0
    
    img = image
    M = H
    mode_c = 'C'
    """
    
    rows = image.shape.as_list()[0]
    columns = image.shape.as_list()[1]
    
    if H[2][0] == 0 and H[2][1] == 0 and H[2][2] == 1:
        if H[0][1] == 0 and H[1][0] == 0:
            transform_func = _transform_metric
        else:
            transform_func = _transform_affine
    else:
        transform_func = _transform_projective
    
    row_slices = []
    for tfr in range(rows):
        col = []
        for tfc in range(columns):
            c, r = transform_func(tfc, tfr, H)
            val = bilinear_interpolation(image, rows, columns, r, c, 0)
            col.append(tf.expand_dims(val, 0))
        col = tf.concat(col, 0)
        col = tf.expand_dims(col, 1)
        row_slices.append(col)
    out = tf.concat(row_slices, 1)
    
    return out

def radon(image, theta = None, circle = True):
    """
    Calculates the radon transform of an image given specified
    projection angles.
    
    Parameters
    ----------
    image : array_like, dtype=float
        Input image. The rotation axis will be located in the pixel with
        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.
    theta : array_like, dtype=float, optional
        Projection angles (in degrees). If `None`, the value is set to
        np.arange(180).
    circle : boolean, optional
        Assume image is zero outside the inscribed circle, making the
        width of each projection (the first dimension of the sinogram)
        equal to ``min(image.shape)``.
    
    Returns
    -------
    radon_image : ndarray
        Radon transform (sinogram).  The tomography rotation axis will lie
        at the pixel index ``radon_image.shape[0] // 2`` along the 0th
        dimension of ``radon_image``.
    """
    # tf.rank does not return the correct value if eager execution is off
    imageShape = image.shape.as_list()
    if len(imageShape) != 2:
        raise ValueError('The input image must be 2D')
    if theta is None:
        theta = list(range(180))
    
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
        diagonal = math.sqrt(2) * max(imageShape)
        pad = [math.ceil(diagonal - s) for s in imageShape]
        new_center = [(s + p) // 2 for s, p in zip(imageShape, pad)]
        old_center = [s // 2 for s in imageShape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = tf.pad(image, pad_width)
    
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
        col = tf.reduce_sum(rotated, 1)
        col = tf.expand_dims(col, 1)
        radon_image_cols.append(col)
        
    radon_image = tf.concat(radon_image_cols, 1)
    return radon_image