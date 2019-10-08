
import tensorflow as tf

class GaussInteger():
    def __init__(self, real, imag):

        # Check input type.
        if (type(real) is not int or type(imag) is not int):
            raise TypeError("Inputs a and b of GaussInteger(a, b) must be"
                            + " ints.")

        # Cast variables to create complex number.
        self.real = tf.dtypes.cast(tf.constant([real]), tf.float32)
        self.imag = tf.dtypes.cast(tf.constant([imag]), tf.float32)
        self.num = tf.complex(self.real, self.imag)
