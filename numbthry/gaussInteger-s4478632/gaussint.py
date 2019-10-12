
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
            re = tf.math.real(result).eval()
            im = tf.math.imag(result).eval()

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
        Adds two instances of this class together (or an int).
        """
        if type(other) is not GaussInteger and type(other) is not int:
            raise TypeError("Operand must be int or GaussInteger")

        if type(other) is GaussInteger:        
            with tf.Session() as sess:
                sum_re = int((self.re + other.re).eval())
                sum_im = int((self.im + other.im).eval())
            return GaussInteger(sum_re, sum_im)
        
        with tf.Session() as sess:
            sum_re = int(self.re.eval() + other)
            sum_im = int(self.im.eval())
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

    def mul(self, other):
        """
        Multiplies two instances of this class together.
        """
        if type(other) is not GaussInteger:
            raise TypeError("Operand must be GaussInteger")
        
        with tf.Session() as sess:
            re = int((self.re * other.re).eval() - (self.im * other.im).eval())
            im = int((self.im * other.re).eval() + (self.re * other.im).eval())
        return GaussInteger(re, im)

    def __mul__(self, other):
        """
        Overload the "*" operator.
        """
        if type(other) is int:
            with tf.Session() as sess:
                real = int(self.re.eval() * other)
                imag = int(self.im.eval() * other)
                return GaussInteger(real, imag)
        return self.mul(other)

    def __rmul__(self, other):
        """
        Overload the "*" operator.
        """
        return self.__mul__(other)

    def __imul__(self, other):
        """
        Overloads the "*=" operator.
        """
        self = self * other
        return self

    def floordiv(self, other):
        """
        Performs the floor division of two numbers.
        """
        if type(other) is not int and type(other) is not GaussInteger:
            raise TypeError("Operand must be int or GaussInteger")

        # Normal integer floor division
        if type(other) is int:

            if other == 0:
                raise ZeroDivisionError("Denominator must be non-zero")
            
            with tf.Session() as sess:
                re = int(self.re.eval() // other)
                im = int(self.im.eval() // other)

            return GaussInteger(re, im)

        # GaussianInteger floor division
        with tf.Session() as sess:
            if other.re.eval() == 0 and other.im.eval() == 0:
                raise ZeroDivisionError("Denominator must be non-zero")

            numerator = (self * other.conjugate()).getNum()
            denominator = int(other.norm().real)

            re = numerator.real // denominator
            im = numerator.imag // denominator

            return GaussInteger(int(re), int(im))

    def __floordiv__(self, other):
        """
        Overload the "//" operator.
        """
        return self.floordiv(other)

    def __ifloor__(self, other):
        """
        Overload the //= operator.
        """
        self = self // other
        return self

    def mod(self, other):
        """
        Calculates the value of self % other.
        """
        return self - other * (self // other)

    def __mod__(self, other):
        """
        Overloads the % operator.
        """
        return self.mod(other)

    def __imod__(self, other):
        """
        Overloads the %= operator.
        """
        self = self % other
        return self

    def divmod(self, other):
        """
        Returns a tuple of (divisor, remainder).
        """
        quotient = (self // other)
        remainder = (self.mod(other))
        return quotient, remainder

    def gcd(self, other):
        """
        Calculates the gcd of two gaussian integers.
        """
        if self.norm().real < other.norm().real:
            return other.gcd(self)

        while True:
            if other.norm() == 0:
                break
            q, r = self.divmod(other)
            self, other = other, r
        
        return self

    def __pos__(self):
        return self

    def xgcd(self, other):
        """
        Performs the extended euclidean algorithm on the input.
        """
        a1 = GaussInteger(1, 0)
        b1 = GaussInteger(0, 0)
        a2 = GaussInteger(0, 0)
        b2 = GaussInteger(1, 0)

        a = self
        b = other

        if b.norm().real > a.norm().real:
            a, b = b, a
            a1, b1, a2, b2 = a2, b2, a1, b1

        while a.norm().real != 0:
            q, r = b // a, b % a
            m, n = b1 - a1 * q, b2 - a2 * q
            b, a, b1, b2, a1, a2 = a, r, a1, a2, m, n

        return b, b1, b2

    def __pow__(self, power):
        with tf.Session() as sess:
            result = tf.math.pow(self.getNum(), power).eval()
        return result

    def isprime(self):
        
        
            
