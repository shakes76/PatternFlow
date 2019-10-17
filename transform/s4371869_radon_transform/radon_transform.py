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

import math
import numpy as np
import tensorflow as tf

from warnings import warn
from skimage._shared.utils import convert_to_float # TODO - only use tf

apply_affine_transform = tf.keras.preprocessing.image.apply_affine_transform

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
    elif type(theta) is np.ndarray:
        theta = tf.convert_to_tensor(theta, dtype=theta.dtype)
    else:
        try:
            theta = tf.convert_to_tensor(theta, dtype=theta.dtype)
        except:
            raise TypeError('The input theta must be a tensor or tensor-like')

    # Set default behavior for preserve_range
    if preserve_range is None and np.issubdtype(image.dtype, np.integer):
        warn('Image dtype is not float. By default radon will assume '
             'you want to preserve the range of your image '
             '(preserve_range=True). In scikit-image 0.18 this behavior will '
             'change to preserve_range=False. To avoid this warning, '
             'explictiely specify the preserve_range parameter.',
             stacklevel=2)
        preserve_range = True

    image = convert_to_float(image, preserve_range)
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float64)

    if circle:
        shape_min = min(image.shape)
        radius = shape_min // 2
        img_shape = np.array(image.shape)
        coords = np.array(np.ogrid[:image.shape[0], :image.shape[1]])
        dist = ((coords - img_shape // 2) ** 2).sum(0)
        outside_reconstruction_circle = dist > radius ** 2
        if np.any(image[outside_reconstruction_circle]):
            warn('Radon transform: image must be zero outside the '
                 'reconstruction circle')
        # Crop image to make it square
        slices = tuple(slice(int(np.ceil(excess / 2)),
                             int(np.ceil(excess / 2) + shape_min))
                       if excess > 0 else slice(None)
                       for excess in (img_shape - shape_min))
        padded_image = image[slices]
    else:
        diagonal = np.sqrt(2) * max(image.shape)
        pad = [int(np.ceil(diagonal - s)) for s in image.shape]
        new_center = [(s + p) // 2 for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = np.pad(image, pad_width, mode='constant',
                              constant_values=0)

    image = tf.convert_to_tensor(image, dtype=image.dtype)
    img_shape = tf.convert_to_tensor(image.shape, dtype=tf.int32)

    if circle:
        shape_min = tf.reduce_min(img_shape)
        radius = shape_min // 2
        xs, ys = tf.meshgrid(tf.range(img_shape[1]), tf.range(img_shape[0]))
        dist = (xs - img_shape[1] // 2) ** 2 + (ys - img_shape[0] // 2) ** 2
        outside_reconstruction_circle = dist > radius ** 2
        if tf.math.reduce_any(image[outside_reconstruction_circle]):
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
    radon_image = np.zeros((padded_image.shape[0], len(theta)))

    for i, angle in enumerate(np.deg2rad(theta)):
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        R = np.array([[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
                      [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
                      [0, 0, 1]])
        rotated = warp(padded_image, R, clip=False)
        radon_image[:, i] = rotated.sum(0)


    # padded_image is always square
    if padded_image.shape[0] != padded_image.shape[1]:
        raise ValueError('padded_image must be a square')
    center = padded_image.shape[0] // 2
    radon_image = tf.zeros((padded_image.shape[0], len(theta)))

    theta = theta * (math.pi / 180.0)
    for i, angle in enumerate(theta):
        cos_a, sin_a = tf.math.cos(tf.cast(angle)), tf.math.sin(tf.cast(angle))
        rotated = apply_affine_transform(padded_image, tx=-center, ty=-center)
        rotated = apply_affine_transform(rotated, theta=angle)
        rotated = apply_affine_transform(rotated, tx=center, ty=center)
        radon_image[:, i] = rotated.sum(0)
    return radon_image

