import tensorflow as tf
tf.InteractiveSession()

class DummyArray(object):
    """Dummy object that just exists to hang __array_interface__ dictionaries
    and possibly keep alive a reference to a base array.
    """
    def __init__(self, interface, base=None):
        self.__array_interface__ = interface
        self.base = base
        
def _maybe_view_as_subclass(original_array, new_array):
    if type(original_array) is not type(new_array):
        # if input was an ndarray subclass and subclasses were OK,
        # then view the result as that subclass.
        new_array = new_array.view(type=type(original_array))
        # Since we have done something akin to a view from original_array, we
        # should let the subclass finalize (if it has it implemented, i.e., is
        # not None).
        if new_array.__array_finalize__:
            new_array.__array_finalize__(original_array)
    return new_array

def as_strided(x, shape=None, strides=None, writeable=True):
    """
    Create a view into the array with the given shape and strides.
    
    Parameters
    ----------
    x : tensor
        A new tensor
    shape : sequence of int, optional
        The shape of the new tensor(conver to narray at begining). Defaults to ``x.shape``.
    strides : sequence of int, optional
        The strides of the new tensor(conver to narray at begining). Defaults to ``x.strides``.
    subok : bool, optional
        .. versionadded:: 1.10
        If True, subclasses are preserved.
    writeable : bool, optional
        .. versionadded:: 1.12
        If set to False, the returned array will always be readonly.
        Otherwise it will be writable if the original array was. It
        is advisable to set this to False if possible (see Notes).
    Returns
    -------
    view : ndarray
    See also
    --------
    broadcast_to: broadcast an array to a given shape.
    reshape : reshape an array.
    Notes
    -----
    ``as_strided`` creates a view into the tensor given the exact strides
    and shape. At the begining of as_strided funciton. I use session.run() 
    function to convert the tensor to the numpy arrays.Then we can do some
    operations on such arrays. 
    It is advisable to always use the original ``x.strides`` when
    calculating new strides to avoid reliance on a contiguous memory
    layout.
    Vectorized write operations on such arrays will typically be unpredictable.
    They may even give different results for small, large, or transposed arrays. 
    Furthermore, arrays created with this function often contain self
    overlapping memory, so that two elements are identical.
    Since writing to these arrays has to be tested and done with great
    care, you may want to use ``writeable=False`` to avoid accidental write
    operations.
    For these reasons it is advisable to avoid ``as_strided`` when
    possible.
    """
    # first convert input to array, possibly keeping subclass
    session = tf.Session()
    x = session.run(x)

    interface = dict(x.__array_interface__)
    if shape is not None:
        interface['shape'] = tuple(shape)
    if strides is not None:
        interface['strides'] = tuple(strides)

    array = session.run(tf.convert_to_tensor(DummyArray(interface, base=x)))
    # The route via `__interface__` does not preserve structured
    # dtypes. Since dtype should remain unchanged, we set it explicitly.
    array.dtype = x.dtype

    view = _maybe_view_as_subclass(x, array)

    if view.flags.writeable and not writeable:
        view.flags.writeable = False

    return view

def view_as_blocks(arr_in, block_shape): 
    """Block view of the input n-dimensional tensor (using re-striding). 
  
     Blocks are non-overlapping views of the input tensor. 
  
     Parameters 
     ---------- 
     arr_in : tensor 
         N-d input tensor. 
     block_shape : tuple 
         The shape of the block. Each dimension must divide evenly into the 
         corresponding dimensions of `arr_in`. 
  
     Returns 
     ------- 
     arr_out : ndarray 
       Block view of the input tensor. 
    """
    if not isinstance(block_shape, tuple): 
        raise TypeError('block needs to be a tuple') 
  
    block_shape = tf.convert_to_tensor(block_shape)
    session = tf.Session()
    block_shape = session.run(block_shape)
  
    session = tf.Session()
    arr_in = session.run(arr_in)

    arr_shape = tf.convert_to_tensor(arr_in.shape)
    session = tf.Session()
    arr_shape = session.run(arr_shape)
  
    if (block_shape <= 0).any(): 
        raise ValueError("'block_shape' elements must be strictly positive") 
    if block_shape.size != arr_in.ndim: 
        raise ValueError("'block_shape' must have the same length " 
    "as 'arr_in.shape'")
  
    if (arr_shape % block_shape).sum() != 0: 
        raise ValueError("'block_shape' is not compatible with 'arr_in'") 

    # -- restride the array to build the block view     
    new_shape = tuple(arr_shape // block_shape) + tuple(block_shape) 
    new_strides = tuple(arr_in.strides * block_shape) + arr_in.strides

    arr_in = tf.convert_to_tensor(arr_in)
    arr_out = as_strided(arr_in, shape=new_shape, strides=new_strides) 
  
    return arr_out 

def block_reduce(image, block_size, func=tf.reduce_sum, cval=0):
    """Down-sample image by applying function to local blocks. 
 
    Parameters 
    ---------- 
    image : tensor
            N-dimensional input tensor(image) 
    block_size : array 
                Array containing down-sampling integer factor along each axis. 
    func : callable 
            Function object which is used to calculate the return value for each 
            local block. This function must implement an ``axis`` parameter such 
            as ``tf.reduce_sum`` or ``tf.reduce_mean``. 
    cval : float 
            Constant padding value if image is not perfectly divisible by the 
            block size. 
    Returns 
    ------- 
    image : ndarray 
            Down-sampled image with same number of dimensions as input image.    
    """ 
    if len(block_size) != image.ndim: 
        raise ValueError("`block_size` must have the same length " 
                        "as `image.shape`.") 
    pad_width = [] 
    for i in range(len(block_size)): 
        if block_size[i] < 1: 
            raise ValueError("Down-sampling factors must be >= 1. Use "
                             "`skimage.transform.resize` to up-sample an " 
                             "image.") 
    if image.shape[i] % block_size[i] != 0: 
        after_width = block_size[i] - (image.shape[i] % block_size[i]) 
    else: 
        after_width = 0
    pad_width.append((0, after_width)) 
   
    t = tf.convert_to_tensor(image)
    image = tf.pad(t, pad_width, "CONSTANT")
  
    blocked = view_as_blocks(image, block_size)
  
    session = tf.Session()
    image = session.run(image)
  
    blocked = tf.convert_to_tensor(blocked)
    blocked = tf.cast(blocked, tf.float64)
    result =  func(blocked, axis=tuple(range(image.ndim, tf.rank(blocked).eval())))

    session = tf.Session()
    result =session.run(result)
 
    return result

def downscale_local_mean(image, factors, cval=0, clip=True):
    """Down-sample N-dimensional image by local averaging. 

    The image is padded with `cval` if it is not perfectly divisible by the 
    integer factors. 

    This function calculates the local mean of elements in each block
    of size `factors` in the input image. 

    Parameters 
    ---------- 
    image : tensor or ndarray 
            There are two situation about the input image
            This function can recieve ndarray and N-dimensional imput tensor image 
    factors : array
            Array containing down-sampling integer factor along each axis. 
    cval : float, optional 
            Constant padding value if image is not perfectly divisible by the 
            integer factors. 
    clip : bool, optional 
            Unused, but kept here for API consistency with the other transforms 
            in this module. (The local mean will never fall outside the range 
            of values in the input image, assuming the provided `cval` also 
            falls within that range.) 
  
    Returns 
    ------- 
    image : ndarray 
            Down-sampled image with same number of dimensions as input image. 
            For integer inputs, the output dtype will be ``float64``. 
    """ 
    
    if tf.is_tensor(image):
        print("tensor")
        session = tf.Session()
        image = session.run(image)
    else:
        print("array")
    return block_reduce(image, factors, tf.reduce_mean, cval)
