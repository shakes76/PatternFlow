import tensorflow as tf


def downscale_local_mean(image, factors, cval=0):
    """
    Down-sample N-dimensional image by local averaging.
    The image is padded with 'cval' if it is not perfectly divisible by the integer factors.
    This function calculates the local mean of elements in each block of size 'factors' in the input image.
    Equivalent to skimage.transform.downscale_local_mean

    :param image: ndarray or tensor
        N-dimensional input image.
    :param factors: array-like
        Array containing down-sampling integer factor along each axis.
    :param cval: float, optional
        Constant padding value if image is not perfectly divisible by the integer factors.

    :return: tensor
        Down-sampled image in the format of tensor with same number of dimensions as input image.

    """

    session = tf.Session()
    # if the input is a tensor, convert it to an ndarray
    if tf.is_tensor(image):
        image = session.run(image)
    image_downscaled = block_reduce(image, factors, tf.reduce_mean, cval)

    return tf.convert_to_tensor(image_downscaled)


def block_reduce(image, block_size, func=tf.reduce_sum, cval=0):
    """
    Down-sample image by applying function to local blocks.
    Equivalent to skimage.measure.block.block_reduce

    :param image: ndarray
        N-dimensional input image.
    :param block_size: array-like
        Array containing down-sampling integer factor along each axis.
    :param func: callable
        Function object which is used to calculate the return value for each local block. This function must implement
        an axis parameter such as tf.reduce_sum or tf.reduce_min
    :param cval: float
        Constant padding value if image is not perfectly divisible by the block size.

    :return: ndarray
        Down-sampled image with same number of dimensions as input image.
    """
    # if the dimension of block_size and image is not the same, raise error
    if len(block_size) != image.ndim:
        raise ValueError("`block_size` must have the same length as `image.shape`.")

    # apply the pad operation on image
    pad_width = []
    for i in range(len(block_size)):
        if block_size[i] < 1:
            raise ValueError("Down-sampling factors must be >= 1.")
        if image.shape[i] % block_size[i] != 0:
            after_width = block_size[i] - (image.shape[i] % block_size[i])
        else:
            after_width = 0
        pad_width.append((0, after_width))
    image = tf.convert_to_tensor(image)
    image = tf.pad(image, pad_width, "CONSTANT")

    # compute the block view of image 
    blocked = view_as_blocks(image, block_size)
    blocked = tf.cast(blocked, tf.float64)
    session = tf.Session()
    image = session.run(image)
    blocked = session.run(blocked)
    # apply the given func on blocked
    result = func(blocked, axis=tuple(range(image.ndim, blocked.ndim)))

    return session.run(result)


def view_as_blocks(arr_in, block_shape):
    """
    Block view of the input n-dimensional array using re-striding. BLocks are non-overlapping views of the input array.
    Equivalent to skimage.measure.shape.view_as_blocks

    :param arr_in: ndarray
        N-dimensional input array
    :param block_shape: tuple
        The shape of the block. Each dimension must divide evenly into the corresponding dimensions of arr_in.

    :return: ndarray
        Block view of the input arr_in.
    """
    # check the type of the input block_shape
    if not isinstance(block_shape, tuple):
        raise TypeError('block needs to be a tuple')
    # convert block shape to array format
    session = tf.Session()
    block_shape = tf.convert_to_tensor(block_shape)
    block_shape = session.run(block_shape)
    # raise errors if the input has problems
    if (block_shape <= 0).any():
        raise ValueError("'block_shape' elements must be strictly positive")
    arr_in = session.run(arr_in)
    if block_shape.size != arr_in.ndim:
        raise ValueError("'block_shape' must have the same length "
                         "as 'arr_in.shape'")
    arr_shape = arr_in.shape
    if (arr_shape % block_shape).sum() != 0:
        raise ValueError("'block_shape' is not compatible with 'arr_in'")
    if not arr_in.flags.contiguous:
        raise ValueError("Cannot provide views on a non-contiguous input "
                         "array without copying.")

    # compute the new shape of the input arr_in
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape)
    # compute the new strides of the input arr_in
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides
    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides)

    return arr_out


