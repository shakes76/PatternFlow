#!/usr/bin/env python
"""Wiener-Hunt Deconvolution, Tensorflow Version
"""
__author__ = "Youwen Mao"
__email__ = "youwen.mao@uq.net.au"
__reference__ = ["scikit-image.skimage.restoration.deconvolution", "scikit-image.skimage.restoration.uft.py"]

import tensorflow as tf

def _ir2tf(imp_resp, shape, sess, dim=None, is_real=True):
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
        if tf.contrib.framework.is_tensor(imp_resp):
            dim = len(imp_resp.shape)
        else:
            dim = imp_resp.ndim
    irpadded = tf.Variable(tf.zeros(shape))
    init_op = tf.variables_initializer([irpadded])
    sess.run(init_op)
    imp_shape = tuple(tf.shape(imp_resp).eval())
    op = tf.assign(irpadded[tuple([slice(0, s) for s in imp_shape])], imp_resp)
    sess.run(op)
    for axis, axis_size in enumerate(imp_shape):
        if axis >= len(imp_resp.shape) - dim:
            irpadded = tf.manip.roll(irpadded,
                               shift=-tf.cast(tf.math.floor(axis_size / 2), tf.int32),
                               axis=axis)
    if is_real:
        if dim == 1:
            return tf.spectral.rfft(irpadded)
        elif dim == 2:
            return tf.spectral.rfft2d(irpadded)
        elif dim == 3:
            return tf.spectral.rfft3d(irpadded)
        else:
            raise ValueError('Bad dimension, dim can only be 1, 2 and 3')
    else:
        if dim == 1:
            return tf.fft(irpadded)
        elif dim == 2:
            return tf.fft2d(irpadded)
        elif dim == 3:
            return tf.fft3d(irpadded)
        else:
            raise ValueError('Bad dimension, dim can only be 1, 2 and 3')

def _laplacian(ndim, shape, sess, is_real=True):
    impr = tf.Variable(tf.zeros([3] * ndim))
    tf.global_variables_initializer().run()
    for dim in range(ndim):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (ndim - dim - 1))

        op = tf.assign(impr[idx], tf.reshape(tf.convert_to_tensor([-1.0,0.0,-1.0]), [-1 if i == dim else 1 for i in range(ndim)]))
        sess.run(op)
    op = tf.assign(impr[[1]*ndim], 2.0 * ndim)
    sess.run(op)
    return _ir2tf(impr, shape, is_real=is_real), impr

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
        reg = _laplacian(image.ndim, image.shape, sess, is_real=is_real)
    if (reg.dtype != tf.complex64) & (reg.dtype != tf.complex128):
        reg = _ir2tf(reg, image.shape, sess, is_real=is_real)
    if psf.shape != reg.shape:
        trans_func = _ir2tf(psf, image.shape, sess, is_real=is_real)
    else:
        trans_func = psf
    
