#!/usr/bin/env python
"""Wiener-Hunt Deconvolution, Tensorflow Version
"""
__author__ = "Youwen Mao"
__email__ = "youwen.mao@uq.net.au"
__reference__ = ["scikit-image.skimage.restoration.deconvolution", "scikit-image.skimage.restoration.uft.py"]

import tensorflow as tf

def _ir2tf(imp_resp, shape, dim=None, is_real=True):
    """Compute the transfer function of an impulse response (IR).
    This function makes the necessary correct zero-padding, zero
    convention, correct fft2, etc... to compute the transfer function
    of IR. To use with unitary Fourier transform for the signal (ufftn
    or equivalent).

    Args:
        imp_resp (ndarray): he impulse responses.
        shape (tuple): A tuple of integer corresponding to the target shape of 
            the transfer function.
        dim (int): The last axis along which to compute the transform. All
            axes by default.
        is_real (boolean): If True (default), imp_resp is supposed real and the
            Hermitian property is used with rfftn Fourier transform.

    Return:
        tensor: The transfer function of shape ``shape``.
    """
    if not dim:
        dim = imp_resp.ndim
    irpadded = tf.zeros(shape)
    irpadded = irpadded.eval()
    m_shape = imp_resp.shape
    x_forward = tf.cast(tf.math.ceil(m_shape[0]/2), tf.int32).eval()
    y_forward = tf.cast(tf.math.ceil(m_shape[1]/2), tf.int32).eval()
    x_backward = (tf.shape(irpadded)[0] 
        - tf.cast(tf.math.floor(m_shape[0]/2), tf.int32)).eval()
    y_backward = (tf.shape(irpadded)[1] 
        - tf.cast(tf.math.floor(m_shape[1]/2), tf.int32)).eval()
    for row_f in range(0, x_forward):
        for col_f in range(0, y_forward):
            irpadded[row_f, col_f] = 1
        for col_b in range(y_backward, shape[0]):
            irpadded[row_f, col_b] = 1
    for row_b in range(x_backward, shape[0]):
        for col_f in range(0, y_forward):
            irpadded[row_b, col_f] = 1
        for col_b in range(y_backward, shape[1]):
            irpadded[row_b, col_b] = 1
    if is_real:
        if dim == 1:
            return tf.spectral.rfft(irpadded)
        elif dim == 2:
            return tf.spectral.rfft2d(irpadded)
        elif dim == 3:
            return tf.spectral.rfft3d(irpadded)
        else:
            raise ValueError('Bad dimension value, dim can only be 1, 2 and 3.')
    else:
        if dim == 1:
            return tf.fft(irpadded)
        elif dim == 2:
            return tf.fft2d(irpadded)
        elif dim == 3:
            return tf.fft3d(irpadded)
        else:
            raise ValueError('Bad dimension value, dim can only be 1, 2 and 3.')

def _laplacian():
    pass

def wiener(image, psf, balance, reg=None, is_real=True, clip=True):
    """Deconvolution with Wiener filter
    
    Args:
        image (ndarray): Input degraded image.
        psf (ndarray): This is assumed to be the impulse response (input image 
            space) if the data-type is real, or the transfer function (Fourier 
            space) if the data-type is complex. There is no constraints on the
            shape of the impulse response. The transfer function must be of 
            shape `(M, N)` if `is_real is True`, `(M, N // 2 + 1)` otherwise
        balance(float): The regularisation parameter value that tunes the 
            balance between the data adequacy that improve frequency 
            restoration and the prior adequacy that reduce frequency 
            restoration.
        reg (tensor): The regularisation operator. The Laplacian by default. 
            It can be an impulse response or a transfer function, as for the
            psf. Shape constraint is the same as for the `psf` parameter.
        is_real (boolean): True by default. Specify if ``psf`` and ``reg`` are 
            provided with hermitian hypothesis, that is only half of the 
            frequency plane is provided (due to the redundancy of Fourier 
            transform of real signal). It's apply only if ``psf`` and/or 
            ``reg`` are provided as transfer function. 
        clip (boolean): True by default. If True, pixel values of the result 
            above 1 or under -1 are thresholded for skimage pipeline 
            compatibility.
    Return:
        ndarray: The predicted original image
    """
    #TF initialization
    sess = tf.InteractiveSession()
    if reg is None:
        reg = _laplacian(image.ndim, image.shape, is_real=is_real)
    if (reg.dtype != tf.complex64) & (reg.dtype != tf.complex128):
        _ir2tf(reg, image.shape, is_real=is_real)
