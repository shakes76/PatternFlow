import tensorflow as tf


def downscale_local_mean(image, factors, cval=0):
    """Down-sample N-dimensional image by local averaging.

    The image is padded with 'cval' if it is not perfectly divisible by the integer factors.

    This function calculates the local mean of elements in each block of size 'factors' in the input image.
    Equivalent to skimage.transform.downscale_local_mean

    Parameter
    ---------
    image : ndarray or tensor
        N-dimensional input image.
    factors : array-like
        Array containing down-sampling integer factor along each axis.
    cval : float, optional
        Constant padding value if image is not perfectly divisible by the integer factors.

    Returns
    -------
    image : tensor
        Down-sampled image in the format of tensor with same number of dimensions as input image.

    """
    session = tf.Session()
    # if the input is a tensor, convert it to an ndarray
    if tf.is_tensor(image):
        image = session.run(image)
    image_downscaled = block_reduce(image, factors, tf.reduce_mean, cval)

    return tf.convert_to_tensor(image_downscaled)

