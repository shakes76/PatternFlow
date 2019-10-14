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

    :param image: (Tensor) Input image
    :param gamma: (float) Non negative real number. Default value is 1
    :param gain: (float) The constant multiplier. Default value is 1
    :return: (ndarray) Gamma corrected output image.
    """
    image = tf.constant(image)
    # check the input image are all non negative
    tf.debugging.assert_non_negative(image).mark_used()
    dtype = image.dtype

    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number.")

    gamma = tf.constant(gamma, dtype=dtype)
    out = tf.pow(image, gamma) * gain
    return out.numpy()


def adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False):
    """Applies Sigmoid Correction on the input image.
    This function adjust the input image according to the
    equation ``O = 1/(1 + exp*(gain*(cutoff - I)))``

    :param image: (Tensor) Input image
    :param cutoff: (float) Cutoff of the sigmoid function that shifts the
                    characteristic curve in horizontal direction.
                    Default value is 0.5.
    :param gain: (float) The constant multiplier in exponential's power of
                    sigmoid function. Default value is 10.
    :param inv: (bool) If True, returns the negative sigmoid correction.
                    Defaults to False.
    :return: (ndarray) Sigmoid corrected output image.
    """
    image = tf.constant(image)
    # check the input image are all non negative
    tf.debugging.assert_non_negative(image).mark_used()
    dtype = image.dtype

    # cutoff = tf.constant(cutoff, dtype=dtype)

    out = 1 / (1 + tf.exp(gain * (cutoff - image)))
    if inv:
        out = 1 - out
    with tf.Session() as sess:
        out = sess.run(out)
    return out
