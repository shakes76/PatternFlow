# Number Theoretic Functions implemented in Tensorflow
#### COMP3170 Pattern Recognition and Analysis (Final Project)

numbthy\_tf.py module requires tensorflow 2.0 to run

numbthy\_tf.py module contains the following number theoretic functions:

    gcd(a,b)            - Compute the greatest common divisor of a and b.
    xgcd(a,b)           - Find [g,x,y] such that g=gcd(a,b) and g = ax + by.
    power_mod(b,e,n)    - Compute b^e mod n efficiently.
    inverse_mod(b,n)    - Compute 1/b mod n.
    is_prime(n)         - Test whether n is prime using a variety of pseudoprime tests.
    isprimeF(n,b)       - Test whether n is prime or a Fermat pseudoprime to base b.
    isprimeE(n,b)       - Test whether n is prime or an Euler pseudoprime to base b.
    factorone(n)        - Find a factor of n using a variety of methods.
    factorPR(n)         - Find a factor of n using the Pollard Rho method.

The original algorithmns are created by Robert-Campbell-256. The files can be found here:
[Robert-Campbell-256/Number-Theory-Python/numbthy.py]
(https://github.com/Robert-Campbell-256/Number-Theory-Python/blob/master/numbthy.py)
Robert-Campbell-256 uses numpy, functools and math module to construct the number theoretics functions. 
Whereas, this module uses the same algorithmns, but in tensorflow version of them. So, to run
this module, users are required to import tensorflow and run a session in it. 

E.g.
```python
import tensorflow as tf
import numbthy_tf as nm

sess = tf.InteractiveSession()

tf.global_variable_initializer().run()

integer_1 = tf.constant(123)
integer_2 = tf.constant(24)
result = nm.gcd(integer_1, integer_2)
print(result.eval())    #it will print 3
```

###Important Note

There are several functions which work differently from Robert-Campbell-256:

isprimeF(n, b):

(Fixed isprimeF. Instead of using tf.math.pow, it is changed to power\_mod)

inverse\_mod(a, n):

If the integer a has no inverse (mod n), the function should raise a ValueError. However, tensorflow does not seem to support
raising an error. So, how this module does is that it returns a tuple with two integers: if the second integer is 0, then the inverse\_mod does not exist and should disregard the first integer; if the second integer is 1, then the first integer will be the result.

power\_mod(b, e, n):

Same as inverse\_mod. disregard the first integer if the second is a 0.

###Test Drive

The test drive will run through every tensorflow implemented function and compare the result with the original functions.
Some tests will have random numbers to test its functionality, but some tests will use predetermine numbers to test.
