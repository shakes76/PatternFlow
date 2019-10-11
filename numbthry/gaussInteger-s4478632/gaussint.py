
import tensorflow as tf

class GaussInteger():
    def __init__(self, real, imag):
        """
        Initialises the class. real is the real part and imag is the
        imaginary part.
        """
        # Check input type.
        if (type(real) is not int or type(imag) is not int):
            raise TypeError("Inputs a and b of GaussInteger(a, b) must be"
                            + " ints.")

        # Cast variables to create complex number.
        self.real = tf.dtypes.cast(tf.constant([real]), tf.float32)
        self.imag = tf.dtypes.cast(tf.constant([imag]), tf.float32)
        self.num = tf.complex(self.real, self.imag)

    def __str__(self):
        """
        Return a string representation of the number.
        """
        return str(self.getNum())[1:-1]

    def getNum(self):
        """
        Returns the complex number as a python complex type.
        """
        with tf.Session() as sess:
            return complex(self.num.eval()[0])

    def conjugate(self):
        """
        Computes and returns a class of the complex conjugate.
        """
        result = tf.math.conj(self.num)

        # Retrieve and evaluate the components.
        with tf.Session() as sess:
            real = tf.math.real(result).eval()
            imag = tf.math.imag(result).eval()

        return GaussInteger(int(real), int(imag))

    def norm(self):
        """
        Calculates and returns the norm as type python complex.
        """
        conj = self.conjugate()

        with tf.Session() as sess:
            return complex(tf.math.multiply(self.num, conj.num).eval())
