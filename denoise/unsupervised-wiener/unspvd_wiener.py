import tensorflow as tf


def ir2tf(imp_resp, shape, dim=None, is_real=True):
    """Compute the transfer function of an impulse response (IR).
    This function makes the necessary correct zero-padding, zero
    convention, correct fft2, etc... to compute the transfer function
    of IR. To use with unitary Fourier transform for the signal (ufftn
    or equivalent).
    Parameters
    ----------
    imp_resp : ndarray
        The impulse responses.
    shape : tuple of int
        A tuple of integer corresponding to the target shape of the
        transfer function.
    dim : int, optional
        The last axis along which to compute the transform. All
        axes by default.
    is_real : boolean, optional
       If True (default), imp_resp is supposed real and the Hermitian property
       is used with rfftn Fourier transform.
    Returns
    -------
    y : complex ndarray
       The transfer function of shape ``shape``.
    See Also
    --------
    ufftn, uifftn, urfftn, uirfftn
    Examples
    --------
    >>> np.all(np.array([[4, 0], [0, 0]]) == ir2tf(np.ones((2, 2)), (2, 2)))
    True
    >>> ir2tf(np.ones((2, 2)), (512, 512)).shape == (512, 257)
    True
    >>> ir2tf(np.ones((2, 2)), (512, 512), is_real=False).shape == (512, 512)
    True
    Notes
    -----
    The input array can be composed of multiple-dimensional IR with
    an arbitrary number of IR. The individual IR must be accessed
    through the first axes. The last ``dim`` axes contain the space
    definition.
    """
    if not dim:
        dim = imp_resp.ndim
    # Zero padding and fill
    irpadded = tf.zeros(shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    # Roll for zero convention of the fft to avoid the phase
    # problem. Work with odd and even size.
    for axis, axis_size in enumerate(imp_resp.shape):
        if axis >= imp_resp.ndim - dim:
            irpadded = tf.roll(irpadded,
                               shift=-int(tf.floor(axis_size / 2)),
                               axis=axis)
    if dim == 1:
        if is_real:
            return tf.signal.rfft(irpadded)
        else:
            return tf.fft(tf.cast(irpadded, tf.complex64))
    elif dim == 2:
        if is_real:
            return tf.signal.rfft2d(irpadded)
        else:
            return tf.fft2d(tf.cast(irpadded, tf.complex64))
    elif dim == 3:
        if is_real:
            return tf.signal.rfft3d(irpadded)
        else:
            return tf.fft3d(tf.cast(irpadded, tf.complex64))
    else:
        raise ValueError('Dimension can only be 1, 2 and 3')


def laplacian(ndim, shape, is_real=True):
    """Return the transfer function of the Laplacian.
    Laplacian is the second order difference, on row and column.
    Parameters
    ----------
    ndim : int
        The dimension of the Laplacian.
    shape : tuple
        The support on which to compute the transfer function.
    is_real : boolean, optional
       If True (default), imp_resp is assumed to be real-valued and
       the Hermitian property is used with rfftn Fourier transform
       to return the transfer function.
    Returns
    -------
    tf : array_like, complex
        The transfer function.
    impr : array_like, real
        The Laplacian.
    Examples
    --------
    >>> tf, ir = laplacian(2, (32, 32))
    >>> np.all(ir == np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]]))
    True
    >>> np.all(tf == ir2tf(ir, (32, 32)))
    True
    """
    impr = tf.zeros([3] * ndim)
    for dim in range(ndim):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (ndim - dim - 1))
        impr[idx] = tf.array([-1.0,
                              0.0,
                              -1.0]).reshape([-1 if i == dim else 1
                                              for i in range(ndim)])
    impr[(slice(1, 2), ) * ndim] = 2.0 * ndim
    return ir2tf(impr, shape, is_real=is_real), impr


def unsupervised_wiener(image, psf, reg=None, user_params=None, is_real=True,
                        clip=True):

    params = {'threshold': 1e-4, 'max_iter': 200,
              'min_iter': 30, 'burnin': 15, 'callback': None}
    params.update(user_params or {})

    if reg is None:
        reg, _ = laplacian(image.ndim, image.shape, is_real=is_real)
    if not is_real:
        reg = ir2tf(reg, image.shape, is_real=is_real)

    if psf.shape != reg.shape:
        trans_fct = ir2tf(psf, image.shape,  is_real=is_real)
    else:
        trans_fct = psf
