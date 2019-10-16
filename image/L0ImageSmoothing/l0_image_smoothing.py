################################################################################
# L0 Gradient Norm Image Smoothing Algorithm
# Implemented in: Tensorflow 2.0
#
# Author: Jameson Nguyen (JNRuan)
################################################################################
# ATTRIBUTIONS
#
# [2]. Alexandre Boucaud, “pypher: Python PSF Homogenization kERnels”. Zenodo, 02-Sep-2016.
#
################################################################################
from imageio import imread
import matplotlib.pyplot as plt
import tensorflow as tf


def _zero_pad_fxypsf(psf, shape):
    """
    Pads point spread function (psf) Fx or Fy with zeroes up to target shape.
    The target shape is the shape of the image we want to smooth.

    Keeps original psf functions in the initial positions of the tensor and pads
    the rest of the tensor indices with zeroes.

    This method is a pre-processing step prior to conversion to an optical
    transfer function (OTF).

    Expects one of two psf functions:
    Fx = [[-1, 1]]
    Fy = [[-1]. [1]]

    :param psf: Tensor containing a point spread function for padding.
    :param shape tuple(int, int): Target shape from image we are smoothing.
    :return: psf function padded with zeroes with new shape=(shape[0], shape[1])
    """
    if psf.shape[0] == 1:
        # PSF is Fx = [[-1, 1]]
        indices = [[0, 0], [0, 1]]
        psf_padded = tf.SparseTensor(indices, psf[0], shape)
        psf_padded = tf.sparse.to_dense(psf_padded, default_value=0)
    elif psf.shape[0] == 2:
        # PSF is Fy = [[-1], [1]]
        indices = [[0, 0], [1, 0]]
        psf_padded = tf.SparseTensor(indices, psf[:, 0], shape)
        psf_padded = tf.sparse.to_dense(psf_padded, default_value=0)
    return psf_padded

def _fxypsf_to_otf(psf, target):
    """
    _fxy_psf2otf is an adapted function specifically for the L0 norm algorithm
    which originally made use of a matlab psf2otf function. This function is
    therefore a port to tensorflow based off the matlab psf2otf function via a
    python port [2], but specifically adapted for the image smoothing algorithm.

    Converts point spread function (psf) to optical transfer function (otf). Using
    the fast fourier transform on a padded psf.

    Expects to work with psf of either:
    Fx = [[-1, 1]], or
    Fy = [[-1], [1]]
    ============================================================================
    Attribution:
    Original function was a numpy port of a matlab psf2otf function. For a general
    use psf2otf function, recommend using the original numpy port.

    Original python implementation: https://github.com/aboucaud/pypher [2]

    :param psf: Tensor containing a point spread function for padding.
    :param target: Target img for smoothing to compute otf up to target dimensions.
    :return: otf of original provided psf functions Fx or Fy.
    """
    target_shape = (target.shape[0], target.shape[1])
    psf_padded = _zero_pad_fxypsf(psf, target_shape)

    # Per matlab implementation, to ensure off-center psf does not later otf,
    # circular shift psf until central pixel is in (0, 0) position.
    for axis, axis_sz in enumerate(psf.shape):
        psf_padded = tf.roll(psf_padded, shift=tf.constant(-axis_sz // 2), axis=axis)

    # Calculate otf
    # Cast psf to complex number per tensorflow spec for 2d fast fourier transform.
    psf_padded = tf.cast(psf_padded, dtype=tf.complex64)
    otf = tf.signal.fft2d(psf_padded)
    return otf


def l0_image_smoothing(img, _lambda=2e-2, kappa=2.0, beta_max=1e5):
    """
    Applies L0 Image Smoothing [1] on target img.

    Lambda is a hyperparameter to tune degree of smoothing.
    By default this is 2e-2, authors recommend a range of [1e-3, 1e-1] [1].
    Usage note: Smaller lambda results in retaining of more of the original details of image.

    Kappa is the scaling factor that scales rate of smoothing,
    smaller kappa scalar results in more iterations and sharper edges.
    Authors recommend range (1, 2].

    Iterations of smoothing based on beta < beta_max.
    With beta initialised as 2 * lambda.
    In addition, beta is incremented at rate beta * kappa each iteration.

    :param img: Input image, read in as numpy array.
    :param _lambda: Smoothing parameter for degree of smoothness [1]. Default 2e-2.
    :param kappa: Scale rate of smoothing. Default 2.0
    :param beta_max: Parameter to scale max iterations, each iteration increments beta * kappa.
    :return:
    """
    # Image needs to be complex64 or complex128 for tensorflows fourier transform.
    img_tensor = tf.convert_to_tensor(img, dtype=tf.complex64)
    psf_fx = tf.constant([[1, -1]], dtype=tf.int8)
    psf_fy = tf.constant([[1], [-1]], dtype=tf.int8)
    otf_fx = _fxypsf_to_otf(psf_fx, img_tensor)
    otf_fy = _fxypsf_to_otf(psf_fy, img_tensor)

    # Norm and Denorm of input per matlab implementation [1], used for the two
    # subproblems (h-v, S) [1].
    norm_input1 = tf.signal.fft2d(img_tensor)
    denorm_input2 = tf.abs(otf_fx)**2 + tf.abs(otf_fy)**2

    # Convert denorm to 3 channels for colour:
    if img.shape[2] > 1:
        denorm_input2 = tf.tile(tf.expand_dims(denorm_input2, 2), [1, 1, 3])

    return

if __name__ == '__main__':
    img = imread('./bengalcat.jpg')
    l0_image_smoothing(img)
