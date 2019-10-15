"""Logarithmic, Gamma and Sigmoid Correction, Tensorflow version
"""
__author__ = "Yufeng Liu"
__email__ = "yufeng.liu1@uq.net.au"
__reference__ = "scikit-image.skimage.exposure.exposure"

import tensorflow as tf


def adjust_log(image, gain=1, inv=False):
    """Applies Logarithmic correction on the input image.
    This function adjust the input image according to the
    equation ``O = gain*log(1 + I)``
    For inverse logarithmic correction, the equation is
    ``O = gain*(2**I - 1)``.

    :param image: (ndarray) Input image
    :param gain: (float) The constant multiplier. Default value is 1.
    :param inv: (bool) If False, it performs logarithmic correction,
                        else correction will be inverse logarithmic. Defaults to False.
    :return: (ndarray) Logarithm corrected output image.
    """
    dtype = image.dtype
    image = tf.constant(image, tf.float32)
    # check the input image are all non negative
    tf.debugging.assert_non_negative(image)
    norm = image / 255
    if inv:
        base = tf.constant(2.0)
        out = (tf.pow(base, norm) - 1) * gain
    else:
        out = tf.math.log1p(norm) * gain / tf.math.log(2.0)
    inv_norm = out * 255
    out = tf.dtypes.cast(inv_norm, dtype)
    return out.numpy()


def adjust_gamma(image, gamma=1, gain=1):
    """Applies Gamma Correction on the input image.
    This function adjust the input image according to the
    equation ``O = I**gamma``.

    :param image: (ndarray) Input image
    :param gamma: (float) Non negative real number. Default value is 1
    :param gain: (float) The constant multiplier. Default value is 1
    :return: (ndarray) Gamma corrected output image.
    """
    dtype = image.dtype
    image = tf.constant(image, tf.float32)
    # check the input image are all non negative
    tf.debugging.assert_non_negative(image)
    norm = image / 255
    # norm = image
    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number.")

    gamma = tf.constant(gamma, tf.float32)
    out = tf.pow(norm, gamma) * gain
    inv_norm = out * 255
    # inv_norm = out
    out = tf.dtypes.cast(inv_norm, dtype)
    return out.numpy()
