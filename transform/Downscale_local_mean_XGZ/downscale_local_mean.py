import tensorflow as tf
print("TF Version: ", tf.__version__)
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
    # first convert input to array, possibly keeping subclass
    x = tf.convert_to_tensor(x)
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

def downscale_local_mean(image, factors, cval=0, clip=True):
  if tf.is_tensor(image):
    print("tensor")
    session = tf.Session()
    image = session.run(image)
  else:
    print("array")
  return block_reduce(image, factors, tf.reduce_mean, cval)
