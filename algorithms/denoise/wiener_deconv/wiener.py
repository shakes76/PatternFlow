#!/usr/bin/env python
"""Wiener Deconvolution, Tensorflow Version
"""
__author__ = "Youwen Mao"
__email__ = "youwen.mao@uq.net.au"
__reference__ = ["scikit-image.skimage.restoration.deconvolution", "scikit-image.skimage.restoration.uft"]

import tensorflow as tf

def _ir2tf(imp_resp, shape, sess, dim=None, is_real=True):
    """Compute the transfer function of an impulse response (IR).
    This function makes the necessary correct zero-padding, zero
    convention, correct fft2, etc... to compute the transfer function
    of IR. To use with unitary Fourier transform for the signal (ufftn
    or equivalent).

    Args:
        imp_resp (ndarray/tensor): he impulse responses.
        shape (tuple): A tuple of integer corresponding to the target shape of 
            the transfer function.
        sess (InteractiveSession): Tensorflow session.
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
    if tf.contrib.framework.is_tensor(imp_resp):
        imp_shape = tuple(tf.shape(imp_resp).eval())
    else:
        imp_shape = imp_resp.shape
    op = tf.assign(irpadded[tuple([slice(0, s) for s in imp_shape])], imp_resp)
    sess.run(op)
    for axis, axis_size in enumerate(imp_shape):
        if axis >= len(imp_resp.shape) - dim:
            irpadded = tf.manip.roll(irpadded,
                               shift=-tf.cast(tf.math.floor(axis_size / 2), 
                               tf.int32),
                               axis=axis)
    if dim == 1:
        return tf.spectral.rfft(irpadded) if is_real else tf.fft(tf.cast(irpadded, tf.complex64))
    elif dim == 2:
        return tf.spectral.rfft2d(irpadded) if is_real else tf.fft2d(tf.cast(irpadded, tf.complex64))
    elif dim == 3:
        return tf.spectral.rfft3d(irpadded) if is_real else tf.fft3d(tf.cast(irpadded, tf.complex64))
    else:
        raise ValueError('Bad dimension, dim can only be 1, 2 and 3')

def _laplacian(ndim, shape, sess, is_real=True):
    """Return the transfer function of the Laplacian.
    Laplacian is the second order difference, on row and column.

    Args:
        ndim (int): The dimension of the Laplacian.
        shape (tuple): The support on which to compute the transfer function.
        sess (InteractiveSession): Tensorflow session.
        is_real (boolean): If True (default), imp_resp is assumed to be 
            real-valued and the Hermitian property is used with rfftn Fourier 
            transform to return the transfer function.

    Returns:
        ndarray: The transfer function.
        ndarray: The Laplacian.
    """
    impr = tf.Variable(tf.zeros([3] * ndim))
    tf.global_variables_initializer().run()
    for dim in range(ndim):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (ndim - dim - 1))

        op = tf.assign(impr[idx], tf.reshape(
            tf.convert_to_tensor([-1.0,0.0,-1.0]), 
            [-1 if i == dim else 1 for i in range(ndim)]))
        sess.run(op)
    op = tf.assign(impr[[1]*ndim], 2.0 * ndim)
    sess.run(op)
    return _ir2tf(impr, shape, sess, is_real=is_real), impr

def wiener(image, psf, balance, reg=None, is_real=True):
    """Deconvolution with Wiener filter
    
    Args:
        image (ndarray): Input degraded image.
        psf (ndarray): Point Spread Function. This is assumed to be the impulse 
            response (input image space) if the data-type is real, or the 
            transfer function (Fourier space) if the data-type is complex. 
            There is no constraints on the shape of the impulse response. 
            The transfer function must be of  shape `(M, N)` if `is_real is 
            True`, `(M, N // 2 + 1)` otherwise
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
    Return:
        ndarray: The predicted original image
    """
    #TF initialization
    sess = tf.InteractiveSession()
    if reg is None:
        reg, _ = _laplacian(image.ndim, image.shape, sess, is_real=is_real)
    if (reg.dtype != tf.complex64) & (reg.dtype != tf.complex128):
        reg = _ir2tf(reg, image.shape, sess, is_real=is_real)
    if psf.shape != reg.shape:
        trans_func = _ir2tf(psf, image.shape, sess, is_real=is_real)
    else:
        trans_func = psf
    wiener_filter = tf.conj(trans_func) / (tf.cast((tf.abs(trans_func) ** 2), trans_func.dtype) +
                                           tf.cast((balance * tf.abs(reg) ** 2), trans_func.dtype))
    if is_real:
        deconv = tf.spectral.irfft2d(wiener_filter * tf.spectral.rfft2d(image))
    else:
        deconv = tf.spectral.ifft2d(wiener_filter * tf.spectral.fft2d(image))
    return deconv.eval()