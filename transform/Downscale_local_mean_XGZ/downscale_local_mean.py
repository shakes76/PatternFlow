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
        

def downscale_local_mean(image, factors, cval=0, clip=True):
  
  return block_reduce(image, factors, tf.reduce_mean, cval)
