import tensorflow as tf


def adjust_log(image, gain=1, inv=False):
    """Performs Logarithmic correction on the input image.
    This function transforms the input image pixelwise according to the
    equation ``O = gain*log(1 + I)`` after scaling each pixel to the range 0 to 1.
    For inverse logarithmic correction, the equation is ``O = gain*(2**I - 1)``.
    Parameters
    ----------
    image : ndarray
        Input image.
    gain : float, optional
        The constant multiplier. Default value is 1.
    inv : float, optional
    If True, it performs inverse logarithmic correction,
        else correction will be logarithmic. Defaults to False.
    Returns
    -------
    out : ndarray
        Logarithm corrected output image.
    See Also
    --------
    adjust_gamma
    References
    ----------
    .. [1] http://www.ece.ucsb.edu/Faculty/Manjunath/courses/ece178W03/EnhancePart1.pdf
    """
    tf.debugging.assert_non_negative(image).mark_used()
    dtype = image.dtype

    if inv:
        base = tf.constant(2, dtype=dtype)
        out = (tf.pow(base, image) - 1) * gain
        return out

    out = tf.math.log(1 + image) * gain
    return out


def adjust_gamma(image, gamma=1, gain=1):
    """Performs Gamma Correction on the input image.
    Also known as Power Law Transform.
    This function transforms the input image pixelwise according to the
    equation ``O = I**gamma`` after scaling each pixel to the range 0 to 1.
    Parameters
    ----------
    image : ndarray
        Input image.
    gamma : float, optional
        Non negative real number. Default value is 1.
    gain : float, optional
        The constant multiplier. Default value is 1.
    Returns
    -------
    out : ndarray
        Gamma corrected output image.
    See Also
    --------
    adjust_log
    Notes
    -----
    For gamma greater than 1, the histogram will shift towards left and
    the output image will be darker than the input image.
    For gamma less than 1, the histogram will shift towards right and
    the output image will be brighter than the input image.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Gamma_correction
    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.moon())
    >>> gamma_corrected = exposure.adjust_gamma(image, 2)
    >>> # Output is darker for gamma > 1
    >>> image.mean() > gamma_corrected.mean()
    True
    """
    tf.debugging.assert_non_negative(image).mark_used()
    dtype = image.dtype

    if gamma < 0:
        raise ValueError("Gamma should be a non-negative real number.")

    gamma = tf.constant(gamma, dtype=dtype)
    out = tf.pow(image, gamma) * gain
    return out


def adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False):
    """Performs Sigmoid Correction on the input image.
    Also known as Contrast Adjustment.
    This function transforms the input image pixelwise according to the
    equation ``O = 1/(1 + exp*(gain*(cutoff - I)))`` after scaling each pixel
    to the range 0 to 1.
    Parameters
    ----------
    image : ndarray
        Input image.
    cutoff : float, optional
        Cutoff of the sigmoid function that shifts the characteristic curve
        in horizontal direction. Default value is 0.5.
    gain : float, optional
        The constant multiplier in exponential's power of sigmoid function.
        Default value is 10.
    inv : bool, optional
        If True, returns the negative sigmoid correction. Defaults to False.
    Returns
    -------
    out : ndarray
        Sigmoid corrected output image.
    See Also
    --------
    adjust_gamma
    References
    ----------
    .. [1] Gustav J. Braun, "Image Lightness Rescaling Using Sigmoidal Contrast
           Enhancement Functions",
           http://www.cis.rit.edu/fairchild/PDFs/PAP07.pdf
    """
    tf.debugging.assert_non_negative(image).mark_used()
    dtype = image.dtype

    # cutoff = tf.constant(cutoff, dtype=dtype)

    out = 1 / (1 + tf.exp(gain * (cutoff - image)))
    if inv:
        out = 1 - out
    return out
