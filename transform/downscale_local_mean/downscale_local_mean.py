import tensorflow as tf


def downscale_local_mean(image, factors, cval=0):
    """
    Down-sample N-dimensional image by local averaging.
    The image is padded with 'cval' if it is not perfectly divisible by the integer factors.
    This function calculates the local mean of elements in each block of size 'factors' in the input image.
    Equivalent to skimage.transform.downscale_local_mean

    :param image: ndarray
        N-dimensional input image.
    :param factors: array-like
        Array containing down-sampling integer factor along each axis.
    :param cval: float, optional
        Constant padding value if image is not perfectly divisible by the integer factors.

    :return: ndarray
        Down-sampled image with same number of dimensions as input image.

    """
    return block_reduce(image, factors, tf.reduce_mean, cval)


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
    result = session.run(result)
    session.close()

    return result


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
    session.close()
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


class DummyArray(object):
    """
    Dummy object that just exists to hang __array_interface__ dictionaries
    and possibly keep alive a reference to a base array.
    """

    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base


def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):
        # if input was an ndarray subclass and subclasses were OK, then view the result as that subclass.
        new_array = new_array.view(type=type(original_array))
        # Since we have done something akin to a view from original_array, we should let the subclass finalize (if it
        # has it implemented, i.e., is not None).
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array


def as_strided(x, shape=None, strides=None, writeable=True):
    """
    Create a view into the array with the given shape and strides.
    Equivalent to numpy.lib.stride_tricks.as_strided

    :param x: ndarray
        Array used to create a view.
    :param shape: sequence of int, optional
        The shape of the new array. Defaults to x.shape.
    :param strides: sequence of int, optional
        The strides of the new array. Defaults to x.strides.
    :param writeable: bool, optional
        If set to False, the returned array will always be readonly. Otherwise it will be writable if the original
        array was. It is advisable to set this to False if possible.

    :return: ndarray
        The created view of the given input array.
    """
    # first convert input to array, possibly keeping subclass
    x = tf.convert_to_tensor(x)
    session = tf.Session()
    x = session.run(x)
    interface = dict(x.__array_interface__)
    # give the given shape and strides information to interface
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)

    array = session.run(tf.convert_to_tensor(DummyArray(interface, base=x)))
    session.close()
    # The route via interface does not preserve structured dtypes. Since dtype should remain unchanged,
    # we set it explicitly.
    array.dtype = x.dtype

    view = _maybe_view_as_subclass(x, array)

    if view.flags.writeable and not writeable:
        view.flags.writeable = False

    return view
