
import tensorflow as tf

class GaussInteger():
    def __init__(self, re, im):
        """
        Initialises the class. real is the real part and imag is the
        imaginary part.
        """
        # Check input type.
        if (type(re) is not int or type(im) is not int):
            raise TypeError("Inputs a and b of GaussInteger(a, b) must be"
                            + " ints.")

        # Cast variables to create complex number.
        self.re = tf.dtypes.cast(tf.constant([re]), tf.float32)
        self.im = tf.dtypes.cast(tf.constant([im]), tf.float32)
        self.num = tf.complex(self.re, self.im)

    def __str__(self):
        """
        Return a string representation of the number.
        """
        return str(self.getNum())[1:-1] if \
                str(self.getNum())[0] == '(' else str(self.getNum())

    def __repr__(self):
        """
        Return a representation of the instance.
        """
        with tf.Session() as sess:
            return "GaussInteger(" + str(int(self.re.eval()[0])) + \
                    ", " + str(int(self.im.eval()[0])) + ")"

    def __eq__(self, other):
        """
        Returns the equality of the two objects. Two instances of this
        class are equivalent if both the real and imaginary parts are
        equal.
        """
        if type(other) is not GaussInteger:
            return False
        else:
            with tf.Session() as sess:
                return (self.re.eval() == other.re.eval() and \
                        self.im.eval() == other.im.eval())[0]

    def __ne__(self, other):
        """
        Returns the negation of the __eq__ method.
        """
        return not self.__eq__(other)

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
            re = tf.math.re(result).eval()
            im = tf.math.im(result).eval()

        return GaussInteger(int(re), int(im))

    def norm(self):
        """
        Calculates and returns the norm as type python complex.
        """
        conj = self.conjugate()

        with tf.Session() as sess:
            return complex(tf.math.multiply(self.num, conj.num).eval())

    def add(self, other):
        """
        Adds two instances of this class together.
        """
        if type(other) is not GaussInteger:
            raise TypeError("Operand must be int or GaussInteger")
        
        with tf.Session() as sess:
            sum_re = int((self.re + other.re).eval())
            sum_im = int((self.im + other.im).eval())

        return GaussInteger(sum_re, sum_im)

    def __add__(self, other):
        """
        Overloads the "+" binary operator.
        """
        if type(other) is int:
            with tf.Session() as sess:
                newRe = int(self.re.eval() + other)
                newIm = int(self.im.eval())
                return GaussInteger(newRe, newIm)
        return self.add(other)

    def __radd__(self, other):
        """
        Overloads the "+" binary operator.
        """
        return self + other

    def __iadd__(self, other):
        """
        Overloads the "+=" operator.
        """
        self = self + other;
        return self

    def __neg__(self):
        """
        Overloads the "-" unary operator.
        """
        with tf.Session() as sess:
            return GaussInteger(-int(self.re.eval()), -int(self.im.eval()))

    def __sub__(self, other):
        """
        Overloads the "-" binary operator.
        """
        return self + (-other)

    def __rsub__(self, other):
        """
        Overloads the "-" binary operator.
        """
        return (-self) + other

    def __isub__(self, other):
        """
        Overloads the "-=" operator.
        """
        self = self - other
        return self

    
