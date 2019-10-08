import tensorflow.compat.v1 as tf


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
    imp_shape = imp_resp.shape
    sess.run(tf.assign(irpadded[tuple([slice(0, s) for s in imp_shape])], imp_resp))
    
    # Roll for zero convention of the fft to avoid the phase
    # problem. Work with odd and even size.
    for axis, axis_size in enumerate(imp_resp.shape):
        if axis >= len(imp_resp.shape) - dim:
            irpadded = tf.roll(irpadded,
                               shift=-tf.cast(tf.floor(tf.cast(axis_size,tf.int32) / 2),tf.int32),
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


def condition(x_postmean, prev_x_postmean, delta, gn_chain, gx_chain, 
              iteration, min_iter, threshold, burnin, areg2, atf2, data_spectrum, trans_fct, 
              image, reg):
    return tf.math.logical_or(
            tf.math.less_equal(iteration, min_iter),
            tf.math.greater_equal(delta, threshold))


def update_x_postmean(x_postmean, prev_x_postmean, x_sample):
    return tf.cast(prev_x_postmean, tf.complex64) + x_sample


def update_delta(x_postmean, prev_x_postmean, iteration, burnin):
    current = x_postmean / tf.cast((iteration - burnin), tf.complex64)
    previous = tf.cast(prev_x_postmean, tf.complex64) / tf.cast((iteration - burnin - 1), tf.complex64)
    return tf.cast(tf.reduce_sum(tf.abs(current - previous)), tf.complex64) / \
            tf.cast(tf.reduce_sum(tf.abs(x_postmean)), tf.complex64) /\
            tf.cast((iteration - burnin), tf.complex64)


def loop_body(x_postmean, prev_x_postmean, delta, gn_chain, gx_chain, 
              iteration, min_iter, threshold, burnin, areg2, atf2, data_spectrum, trans_fct, 
              image, reg):
    # Sample of Eq. 27 p(circX^k | gn^k-1, gx^k-1, y).
        
        # weighting (correlation in direct space)
        precision = gn_chain[-1] * atf2 + gx_chain[-1] * areg2  # Eq. 29
        excursion = tf.cast(tf.sqrt(0.5),tf.complex64) / tf.cast(tf.sqrt(precision),
                tf.complex64) * (tf.cast(tf.random.normal(data_spectrum.shape),
                                 tf.complex64) + 1j * tf.cast(tf.random.normal(data_spectrum.shape),tf.complex64))

        # mean Eq. 30 (RLS for fixed gn, gamma0 and gamma1 ...)
        wiener_filter = tf.cast(gn_chain[-1],tf.complex64) * tf.math.conj(trans_fct) / tf.cast(precision,tf.complex64)

        # sample of X in Fourier space
        x_sample = wiener_filter * data_spectrum + excursion

        # sample of Eq. 31 p(gn | x^k, gx^k, y)
        new_gn_chain = tf.random.gamma(shape=[1],
                                        alpha=[tf.size(image) / 2], 
                                        beta = image_quad_norm(data_spectrum - x_sample * trans_fct))
        #gn_chain.append()

        # sample of Eq. 31 p(gx | x^k, gn^k-1, y)
        new_gx_chain = tf.random.gamma(shape=[1],alpha=[tf.size(image) / 2],
                                        beta=image_quad_norm(x_sample * reg))
        #gx_chain.append()

        # current empirical average
        
        x_postmean = tf.cond(tf.math.greater(iteration, burnin), 
                            lambda: update_x_postmean(
                                    x_postmean, prev_x_postmean, x_sample), 
                                    lambda: tf.cast(x_postmean, tf.complex64))

        delta = tf.cond(tf.math.greater(iteration, (burnin + 1)), 
                         lambda: update_delta(x_postmean, 
                                              prev_x_postmean, 
                                              iteration, burnin), 
                                              lambda: tf.cast(delta, 
                                                              tf.complex64))
        prev_x_postmean = x_postmean
        
        #iteration + 1
        return [x_postmean, prev_x_postmean, delta, gn_chain+[new_gn_chain], gx_chain+[new_gx_chain], 
              iteration+1, min_iter, threshold, burnin, areg2, atf2, data_spectrum, trans_fct, 
              image, reg]
        
    

def unsupervised_wiener(image, psf, reg=None, user_params=None, is_real=True,
                        clip=True):
    sess = tf.InteractiveSession()
    params = {'threshold': tf.constant(1e-4), 'max_iter': 200,
              'min_iter': 30, 'burnin': 15, 'callback': None}
    params.update(user_params or {})

    if reg is None:
        reg, _ = laplacian(image.ndim, image.shape, sess, is_real=is_real)
    if not is_real:
        reg = ir2tf(reg, image.shape, is_real=is_real)

    if psf.shape != reg.shape:
        trans_fct = ir2tf(psf, image.shape, sess, is_real=is_real)
    else:
        trans_fct = psf

    # The mean of the object
    x_postmean = tf.cast(tf.Variable(tf.zeros(trans_fct.shape)), tf.complex64)
    # The previous computed mean in the iterative loop
    prev_x_postmean = tf.cast(tf.Variable(tf.zeros(trans_fct.shape)), tf.complex64)

    # Difference between two successive mean
    delta = tf.Variable(1e-8)
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
        data_spectrum = tf.fft2d(tf.cast(image,tf.complex64))
    iteration = 0
    
    # Gibbs sampling
    loop_vars = [x_postmean, prev_x_postmean, delta, gn_chain, 
                         gx_chain, iteration, params['min_iter'], params['threshold'], 
                         params['burnin'], areg2, atf2, 
                         data_spectrum, trans_fct, image, reg]
    #vars_shape = tf.shape(tf.constant(loop_vars))
    
    loop = tf.while_loop(condition, 
                         loop_body, 
                         loop_vars=loop_vars, 
                         
                         maximum_iterations = params['max_iter'], 
                         parallel_iterations = 1,
                         return_same_structure = True)
    sess.run(tf.global_variables_initializer())
    result = sess.run(loop)
    
    # Empirical average \approx POSTMEAN Eq. 44
    x_postmean = result[0] / (result[5] - params['burnin'])
    if is_real:
        x_postmean = tf.signal.irfft2d(x_postmean)
    else:
        x_postmean = tf.signal.ifft2d(x_postmean)
    sess.close()
    return (x_postmean.eval(), {'noise': result[3], 'prior': result[4]})
