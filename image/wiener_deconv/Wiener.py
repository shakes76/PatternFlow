#!/usr/bin/env python
"""Wiener-Hunt Deconvolution, Tensorflow Version
"""
__author__ = "Youwen Mao"
__email__ = "youwen.mao@uq.net.au"
__reference__ = "scikit-image.skimage.restoration.deconvolution"

import tensorflow as tf

def ir2tf(imp_resp, shape, dim=None, is_real=True):
    pass

def laplacian():
    pass

def wiener(image, psf, balance, reg=None, is_real=True, clip=True):
    """Deconvolution with Wiener filter
    
    Args:
        image (ndarray): Input degraded image
        psf (ndarray): This is assumed to be the impulse response (input image 
            space) if the data-type is real, or the transfer function (Fourier 
            space) if the data-type is complex. There is no constraints on the
            shape of the impulse response. The transfer function must be of 
            shape `(M, N)` if `is_real is True`, `(M, N // 2 + 1)` otherwise
        balance(float): The regularisation parameter value that tunes the 
            balance between the data adequacy that improve frequency 
            restoration and the prior adequacy that reduce frequency 
            restoration.
        reg (ndarray): The regularisation operator. The Laplacian by default. 
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
    train = tf.Variable(X_train)
    test = tf.Variable(X_test)
    tf.global_variables_initializer().run()