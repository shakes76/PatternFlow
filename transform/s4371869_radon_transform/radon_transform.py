#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tensorflow version of the Radom Transform. 

References:
[1] https://scikit-image.org/docs/stable/auto_examples/transform/plot_radon_transform.html#sphx-glr-auto-examples-transform-plot-radon-transform-py
[2] https://www.clear.rice.edu/elec431/projects96/DSP/bpanalysis.html
"""

__author__ = "Ting-Chen Shang"
__email__ = "tingchen.shang@uq.net.au"

# __all__ = ['radon']

import math
import tensorflow as tf

from warnings import warn

def _coord_map(dim, coord, mode):
    """Wrap a coordinate, according to a given mode.

    Parameters
    ----------
    dim : int
        Maximum coordinate.
    coord : int
        Coord provided by user.  May be < 0 or > dim.
    mode : {'W', 'S', 'R', 'E'}
        Whether to wrap, symmetric reflect, reflect or use the nearest
        coordinate if `coord` falls outside [0, dim).
    """
    cmax = dim - 1
    if mode == 'S': # symmetric
        if coord < 0:
            coord = -coord - 1
        if coord > cmax:
            if (coord / dim) % 2 != 0:
                return cmax - (coord % dim)
            else:
                return coord % dim
    elif mode == 'W': # wrap
        if coord < 0:
            return cmax - ((-coord - 1) % dim)
        elif coord > cmax:
            return coord % dim
    elif mode == 'E': # edge
        if coord < 0:
            return 0
        elif coord > cmax:
            return cmax
    elif mode == 'R': # reflect (mirror)
        if dim == 1:
            return 0
        elif coord < 0:
            # How many times times does the coordinate wrap?
            if (-coord / cmax) % 2 != 0:
                return cmax - (-coord % cmax)
            else:
                return -coord % cmax
        elif coord > cmax:
            if (coord / cmax) % 2 != 0:
                return cmax - (coord % cmax)
            else:
                return coord % cmax
    return coord

def _get_pixel2d_tf(image, rows, cols, r, c, mode, cval):
    """Get a pixel from the image, taking wrapping mode into consideration.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : int
        Position at which to get the pixel.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    value : numeric
        Pixel value at given position.

    """
    if mode == 'C':
        if (r < 0) or (r >= rows) or (c < 0) or (c >= cols):
            return cval
        else:
            return image[tf.cast(r * cols + c, tf.int32)]
    else:
        return image[tf.cast(_coord_map(rows, r, mode) * cols \
                     + _coord_map(cols, c, mode), tf.int32)]

def _transform_metric(x, y, H):
    """Apply a metric transformation to a coordinate.

    Parameters
    ----------
    x, y : np_floats
        Input coordinate.
    H : (3,3) *np_floats
        Transformation matrix.
    x_, y_ : *np_floats
        Output coordinate.

    """
    x_ = H[0] * x + H[2]
    y_ = H[4] * y + H[5]

    return (x_, y_)

def _transform_affine(x, y, H):
    """Apply an affine transformation to a coordinate.

    Parameters
    ----------
    x, y : np_floats
        Input coordinate.
    H : (3,3) *np_floats
        Transformation matrix.
    x_, y_ : *np_floats
        Output coordinate.

    """
    x_ = H[0] * x + H[1] * y + H[2]
    y_ = H[3] * x + H[4] * y + H[5]

    return (x_, y_)

def _transform_projective(x, y, H):
    """Apply a homography to a coordinate.

    Parameters
    ----------
    x, y : np_floats
        Input coordinate.
    H : (3,3) *np_floats
        Transformation matrix.
    x_, y_ : *np_floats
        Output coordinate.

    """
    z_ = H[6] * x + H[7] * y + H[8]
    x_ = (H[0] * x + H[1] * y + H[2]) / z_
    y_ = (H[3] * x + H[4] * y + H[5]) / z_

    return (x_, y_)

def _nearest_neighbour_interpolation_tf(image, rows, cols, r, c, mode, cval):
    """Nearest neighbour interpolation at a given position in the image.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : np_float
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    value : np_float
        Interpolated value.

    """
    return _get_pixel2d_tf(image, rows, cols, tf.math.round(r), tf.math.round(c),
                       mode, cval)

def _bilinear_interpolation_tf(image, rows, cols, r, c, mode, cval):
    """Bilinear interpolation at a given position in the image.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : np_float
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    value : numeric
        Interpolated value.

    """
    minr = tf.math.floor(r)
    minc = tf.math.floor(c)
    maxr = tf.math.ceil(r)
    maxc = tf.math.ceil(c)
    dr = r - minr
    dc = c - minc

    top_left = _get_pixel2d_tf(image, rows, cols, minr, minc, mode, cval)
    top_right = _get_pixel2d_tf(image, rows, cols, minr, maxc, mode, cval)
    bottom_left = _get_pixel2d_tf(image, rows, cols, maxr, minc, mode, cval)
    bottom_right = _get_pixel2d_tf(image, rows, cols, maxr, maxc, mode, cval)

    top = (1 - dc) * top_left + dc * top_right
    bottom = (1 - dc) * bottom_left + dc * bottom_right

    return (1 - dr) * top + dr * bottom

def _cubic_interpolation(x, f):
    """Cubic interpolation.

    Parameters
    ----------
    x : np_float
        Position in the interval [0, 1].
    f : real numeric[4]
        Function values at positions [-1, 0, 1, 2].

    Returns
    -------
    value : np_float
        Interpolated value to be used in bicubic_interpolation.

    """
    return (\
        f[1] + 0.5 * x * \
            (f[2] - f[0] + x * \
                (2.0 * f[0] - 5.0 * f[1] + 4.0 * f[2] - f[3] + x * \
                    (3.0 * (f[1] - f[2]) + f[3] - f[0]))))

def _bicubic_interpolation_tf(image, rows, cols, r, c, mode, cval):
    """Bicubic interpolation at a given position in the image.

    Interpolation using Catmull-Rom splines, based on the bicubic convolution
    algorithm described in [1]_.

    Parameters
    ----------
    image : numeric array
        Input image.
    rows, cols : int
        Shape of image.
    r, c : np_float
        Position at which to interpolate.
    mode : {'C', 'W', 'S', 'E', 'R'}
        Wrapping mode. Constant, Wrap, Symmetric, Edge or Reflect.
    cval : numeric
        Constant value to use for constant mode.

    Returns
    -------
    out : np_real_numeric
        Interpolated value.

    References
    ----------
    .. [1] R. Keys, (1981). "Cubic convolution interpolation for digital image
           processing". IEEE Transactions on Signal Processing, Acoustics,
           Speech, and Signal Processing 29 (6): 1153–1160.

    """

    r0 = tf.math.floor(r)
    c0 = tf.math.floor(c)

    # scale position to range [0, 1]
    xr = r - r0
    xc = c - c0

    r0 -= 1
    c0 -= 1
    
    fc = [None] * 4
    fr = [None] * 4

    # row-wise cubic interpolation
    for pr in range(4):
        for pc in range(4):
            fc[pc] = _get_pixel2d_tf(image, rows, cols, pr + r0,
                                     pc + c0, mode, cval)
        fr[pr] = _cubic_interpolation(xc, tf.convert_to_tensor(fc))

    return _cubic_interpolation(xr, tf.convert_to_tensor(fr))

def _warp_fast_tf(image, H, output_shape=None, order=1, mode='constant', cval=0):
    """Projective transformation (homography).

    Perform a projective transformation (homography) of a floating
    point image (single or double precision), using interpolation.

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
    image : 2-D array
        Input image.
    H : array of shape ``(3, 3)``
        Transformation matrix H that defines the homography.
    output_shape : tuple (rows, cols), optional
        Shape of the output image generated (default None).
    order : {0, 1, 2, 3}, optional
        Order of interpolation::
        * 0: Nearest-neighbor
        * 1: Bi-linear (default)
        * 2: Bi-quadratic
        * 3: Bi-cubic
    mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according
        to the given mode.  Modes match the behaviour of `numpy.pad`.
    cval : string, optional (default 0)
        Used in conjunction with mode 'C' (constant), the value
        outside the image boundaries.

    Notes
    -----
    Modes 'reflect' and 'symmetric' are similar, but differ in whether the edge
    pixels are duplicated during the reflection.  As an example, if an array
    has values [0, 1, 2] and was padded to the right by four values using
    symmetric, the result would be [0, 1, 2, 2, 1, 0, 0], while for reflect it
    would be [0, 1, 2, 1, 0, 1, 2].

    """
    M = tf.reshape(H, (-1,))

    dtype = tf.int32 if image.dtype == tf.int32 else tf.int64

    if mode not in ('constant', 'wrap', 'symmetric', 'reflect', 'edge'):
        raise ValueError("Invalid mode specified.  Please use `constant`, "
                         "`edge`, `wrap`, `reflect` or `symmetric`.")
    mode_c = mode[0].upper()

    if output_shape is None:
        out_r = int(image.shape[0])
        out_c = int(image.shape[1])
    else:
        out_r = int(output_shape[0])
        out_c = int(output_shape[1])

    out = tf.zeros((out_r, out_c), dtype=dtype)

    rows = image.shape[0]
    cols = image.shape[1]

    if H[2, 0] == 0 and H[2, 1] == 0 and H[2, 2] == 1:
        if H[0, 1] == 0 and H[1, 0] == 0:
            transform_func = _transform_metric
        else:
            transform_func = _transform_affine
    else:
        transform_func = _transform_projective

    if order == 0:
        interp_func = _nearest_neighbour_interpolation_tf
    elif order == 1:
        interp_func = _bilinear_interpolation_tf
    elif order == 2:
        raise ValueError("Unsupported interpolation order", order)
    elif order == 3:
        interp_func = _bilinear_interpolation_tf
    else:
        raise ValueError("Unsupported interpolation order", order)

    for tfr in range(out_r):
        for tfc in range(out_c):
            c, r = transform_func(tfc, tfr, M)
            out[tfr, tfc] = interp_func(image, rows, cols, r, c, mode_c, cval)

    return out


def radon(image, theta=None, circle=True, *, preserve_range=None):
    """
    Calculates the radon transform of an image given specified
    projection angles.

    Parameters
    ----------
    image : array_like
        Input image. The rotation axis will be located in the pixel with
        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.
    theta : array_like, optional
        Projection angles (in degrees). If `None`, the value is set to
        np.arange(180).
    circle : boolean, optional
        Assume image is zero outside the inscribed circle, making the
        width of each projection (the first dimension of the sinogram)
        equal to ``min(image.shape)``.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html

    Returns
    -------
    radon_image : ndarray
        Radon transform (sinogram).  The tomography rotation axis will lie
        at the pixel index ``radon_image.shape[0] // 2`` along the 0th
        dimension of ``radon_image``.

    References
    ----------
    .. [1] AC Kak, M Slaney, "Principles of Computerized Tomographic
           Imaging", IEEE Press 1988.
    .. [2] B.R. Ramesh, N. Srinivasa, K. Rajgopal, "An Algorithm for Computing
           the Discrete Radon Transform With Some Applications", Proceedings of
           the Fourth IEEE Region 10 International Conference, TENCON '89, 1989

    Notes
    -----
    Based on code of scikit-image
    (https://github.com/scikit-image/scikit-image/blob/de42b4cf11b2a5b5a9e77c54f90bff539947ef0d/skimage/transform/radon_transform.py)

    """
    # Verify image dimension
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    
    # Type checking theta
    if theta is None:
        theta = tf.range(180)
    elif tf.is_tensor(theta):
        pass
    else:
        try:
            theta = tf.convert_to_tensor(theta, dtype=theta.dtype)
        except:
            raise TypeError('The input theta must be a tensor or tensor-like')
    theta = theta * (math.pi / 180.0)

    # Set default behavior for preserve_range
    if preserve_range is None and (image.dtype is not tf.float16 \
            and image.dtype is not tf.float32 \
            and image.dtype is not tf.float64):
        warn('Image dtype is not float. By default radon will assume '
             'you want to preserve the range of your image '
             '(preserve_range=True). In scikit-image 0.18 this behavior will '
             'change to preserve_range=False. To avoid this warning, '
             'explictiely specify the preserve_range parameter.',
             stacklevel=2)
        preserve_range = True

    image = tf.convert_to_tensor(image, dtype=image.dtype)
    img_shape = tf.convert_to_tensor(image.shape, dtype=tf.int32)

    if circle:
        shape_min = tf.reduce_min(img_shape)
        radius = shape_min // 2
        xs, ys = tf.meshgrid(tf.range(img_shape[1]), tf.range(img_shape[0]))
        dist = (xs - img_shape[1] // 2) ** 2 + (ys - img_shape[0] // 2) ** 2
        outside_reconstruction_circle = dist > radius ** 2
        if tf.math.reduce_any(image[outside_reconstruction_circle] > 0.0):
            warn('Radon transform: image must be zero outside the '
                 'reconstruction circle')
        # Crop image to make it square
        slices = tuple(slice(tf.cast(tf.math.ceil(excess / 2), tf.int32),
                             tf.cast(tf.math.ceil(excess / 2), tf.int32) \
                                    + shape_min)
                       if excess > 0 else slice(None)
                       for excess in (img_shape - shape_min))
        padded_image = image[slices]
    else:
        diagonal = tf.math.sqrt(2.0) \
            * tf.cast(tf.reduce_max(img_shape), tf.float32)
        pad_fn = lambda s: \
            tf.cast(tf.math.ceil(diagonal - tf.cast(s, tf.float32)), tf.int32)
        pad = tf.map_fn(pad_fn, image.shape)
        new_center = tf.map_fn(lambda z: tf.math.reduce_sum(z) // 2,
            tf.stack([image.shape, pad], axis=1))   # stack on axis 1 is zip
        old_center = tf.map_fn(lambda s: s // 2, image.shape)
        pad_before = tf.map_fn(lambda cs: tf.reduce_sum(cs),
            tf.stack([-old_center, new_center], axis=1))
        pad_width_fn = lambda ps: tf.convert_to_tensor((ps[0], ps[1] - ps[0]))
        pad_width = tf.map_fn(pad_width_fn,
            tf.stack([pad_before, pad], axis=1))
        padded_image = tf.pad(image, pad_width, mode='constant',
                              constant_values=0)

    # padded_image is always square
    if padded_image.shape[0] != padded_image.shape[1]:
        raise ValueError('padded_image must be a square')
    center = padded_image.shape[0] // 2
    radon_image = tf.zeros((padded_image.shape[0], len(theta)))

    for i, angle in enumerate(theta):
        _angle = tf.cast(angle, tf.float32)
        cos_a, sin_a = tf.math.cos(_angle), tf.math.sin(_angle)
        R = tf.convert_to_tensor([
            [cos_a, sin_a, -center * (cos_a + sin_a - 1)],
            [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
            [0, 0, 1]
        ])
        rotated = _warp_fast_tf(padded_image, R)
        radon[:, i] = tf.reduce_sum(rotated, axis=0)
    return radon_image

