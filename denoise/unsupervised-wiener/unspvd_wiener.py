import tensorflow as tf


def ir2tf(imp_resp, shape, sess, dim=None, is_real=True):
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
        dim = len(imp_resp.shape)
    # Zero padding and fill
    irpadded = tf.Variable(tf.zeros(shape))
    sess.run(tf.variables_initializer([irpadded]))
    #imp_shape = imp_resp.shape
    sess.run(tf.assign(irpadded[tuple([slice(0, s) for s in imp_resp.shape])], imp_resp))

    # Roll for zero convention of the fft to avoid the phase
    # problem. Work with odd and even size.
    for axis, axis_size in enumerate(imp_resp.shape):
        if axis >= len(imp_resp.shape) - dim:
            irpadded = tf.roll(irpadded,
                               shift=-tf.cast(tf.floor(tf.cast(axis_size,tf.int32) / 2),tf.int32),
                               axis=axis)
    if is_real:
        return tf.signal.rfft2d(irpadded)
    else:
        return tf.fft2d(tf.cast(irpadded, tf.complex64))


def laplacian(ndim, shape, sess, is_real=True):
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
    impr = tf.Variable(tf.zeros([3] * ndim))
    sess.run(tf.variables_initializer([impr]))
    for dim in range(ndim):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (ndim - dim - 1))
        assign_op = tf.assign(impr[idx], tf.reshape(
            tf.convert_to_tensor([-1.0, 0.0, -1.0]),
            [-1 if i == dim else 1 for i in range(ndim)]))
        sess.run(assign_op)
    assign_op = tf.assign(impr[[1]*ndim], 2.0 * ndim)
    sess.run(assign_op)
    return ir2tf(impr, shape, sess, is_real=is_real), impr


def image_quad_norm(inarray):
    """Return the quadratic norm of images in Fourier space.
    This function detects whether the input image satisfies the
    Hermitian property.
    Parameters
    ----------
    inarray : ndarray
        Input image. The image data should reside in the final two
        axes.
    Returns
    -------
    norm : float
        The quadratic norm of ``inarray``.
    Examples
    --------
    >>> input = np.ones((5, 5))
    >>> image_quad_norm(ufft2(input)) == np.sum(np.abs(input)**2)
    True
    >>> image_quad_norm(ufft2(input)) == image_quad_norm(urfft2(input))
    True
    """
    # If there is a Hermitian symmetry
    if inarray.shape[-1] != inarray.shape[-2]:
        return (2 * tf.reduce_sum(tf.reduce_sum(tf.abs(inarray) ** 2, axis=-1), axis=-1) -
                tf.reduce_sum(tf.abs(inarray[..., 0]) ** 2, axis=-1))
    else:
        return tf.reduce_sum(tf.reduce_sum(tf.abs(inarray) ** 2, axis=-1), axis=-1)


def unsupervised_wiener(image, psf, reg=None, user_params=None, is_real=True,
                        clip=True):
    sess = tf.InteractiveSession()
    params = {'threshold': 1e-4, 'max_iter': 200,
              'min_iter': 30, 'burnin': 15, 'callback': None}
    params.update(user_params or {})

    if reg is None:
        reg, _ = laplacian(image.ndim, image.shape, sess, is_real=is_real)
    if reg.dtype != tf.complex64:
        reg = ir2tf(reg, image.shape, sess, is_real=is_real)

    if psf.shape != reg.shape:
        trans_fct = ir2tf(psf, image.shape, sess, is_real=is_real)
    else:
        trans_fct = psf

    # The mean of the object
    x_postmean = tf.Variable(tf.cast(tf.zeros(trans_fct.shape), tf.complex64))
    sess.run(tf.variables_initializer([x_postmean]))
    # The previous computed mean in the iterative loop
    prev_x_postmean = tf.Variable(tf.cast(tf.zeros(trans_fct.shape), tf.complex64))
    sess.run(tf.variables_initializer([prev_x_postmean]))

    # Difference between two successive mean
    delta = tf.Variable(1e-8)
    sess.run(tf.variables_initializer([delta]))
    # Initial state of the chain
    gn_chain, gx_chain = [tf.constant(1.0)], [tf.constant(1.0)]

    # The correlation of the object in Fourier space (if size is big,
    # this can reduce computation time in the loop)
    areg2 = tf.abs(reg) ** 2
    atf2 = tf.abs(trans_fct) ** 2

    # The Fourier transform may change the image.size attribute, so we
    # store it.
    if is_real:
        data_spectrum = tf.signal.rfft2d(tf.cast(image,tf.float32))
    else:
        data_spectrum = tf.fft2d(tf.cast(image,tf.float32))

    # Gibbs sampling
    update_prev_x_postmean_op = tf.assign(prev_x_postmean, x_postmean)
    
    bool_op = tf.cond(delta<params['threshold'],lambda:True,lambda:False)
    
    precision = tf.Variable(gn_chain[-1] * atf2 + gx_chain[-1] * areg2)
    sess.run(tf.variables_initializer([precision]))
    
    
    excursion = tf.Variable(tf.cast(tf.sqrt(0.5) / tf.sqrt(precision),tf.complex64) * (tf.cast(tf.random.normal(data_spectrum.shape),tf.complex64) + 1j * tf.cast(tf.random.normal(data_spectrum.shape), tf.complex64)))
    sess.run(tf.variables_initializer([excursion]))
    
    
    wiener_filter = tf.Variable(tf.cast(gn_chain[-1],tf.complex64) * tf.math.conj(trans_fct) / tf.cast(precision,tf.complex64))
    sess.run(tf.variables_initializer([wiener_filter]))
    
    
    x_sample = tf.Variable(wiener_filter * data_spectrum + excursion)
    sess.run(tf.variables_initializer([x_sample]))
    
    update_precision_op = tf.assign(precision, gn_chain[-1] * atf2 + gx_chain[-1] * areg2)
    
    update_excursion_op = tf.assign(excursion, tf.cast(tf.sqrt(0.5) / tf.sqrt(precision),tf.complex64) * (tf.cast(tf.random.normal(data_spectrum.shape),tf.complex64) + 1j * tf.cast(tf.random.normal(data_spectrum.shape), tf.complex64)))
    
    update_wiener_filter_op = tf.assign(wiener_filter, tf.cast(gn_chain[-1],tf.complex64) * tf.math.conj(trans_fct) / tf.cast(precision,tf.complex64))
    sess.run(tf.variables_initializer([wiener_filter]))
    
    update_x_sample_op = tf.assign(x_sample, wiener_filter * data_spectrum + excursion)
    
    update_group = tf.group([update_precision_op,update_excursion_op,update_wiener_filter_op,update_x_sample_op])
    
    for iteration in range(params['max_iter']):
        # Sample of Eq. 27 p(circX^k | gn^k-1, gx^k-1, y).

        # weighting (correlation in direct space)
        sess.run(update_group)
        # sample of X in Fourier space
        if params['callback']:
            params['callback'](x_sample)

        # sample of Eq. 31 p(gn | x^k, gx^k, y)
        gn_chain.append(tf.random.gamma(shape=[1],alpha=[int(image.size / 2)], beta=image_quad_norm(data_spectrum - x_sample * trans_fct)))

        # sample of Eq. 31 p(gx | x^k, gn^k-1, y)
        gx_chain.append(tf.random.gamma(shape=[1],alpha=[int(image.size / 2)],
                                        beta=image_quad_norm(x_sample * reg)))

        # current empirical average
        if iteration > params['burnin']:
            update_x_postmean_op = tf.assign(x_postmean, prev_x_postmean + x_sample)
            sess.run(update_x_postmean_op)

        if iteration > (params['burnin'] + 1):
            current = x_postmean / (iteration - params['burnin'])
            previous = prev_x_postmean / (iteration - params['burnin'] - 1)
            update_delta_op = tf.assign(delta, (tf.reduce_sum(tf.abs(current - previous)) / \
                tf.reduce_sum(tf.abs(x_postmean)) / (iteration - params['burnin'])))
            sess.run(update_delta_op)
        
        sess.run(update_prev_x_postmean_op)
        result = sess.run(bool_op)
        if (iteration > params['min_iter']) and result:
            break

    # Empirical average \approx POSTMEAN Eq. 44
    x_postmean = x_postmean / (iteration - params['burnin'])
    if is_real:
        x_postmean = tf.signal.irfft2d(x_postmean)
    else:
        x_postmean = tf.signal.ifft2d(x_postmean)
    
    return (x_postmean.eval(), {'noise': gn_chain, 'prior': gx_chain})