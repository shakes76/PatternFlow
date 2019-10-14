import tensorflow as tf
import math
import functools


"""start of gcd function"""

def cond_gcd(a,b):
    """An auxiliary function for gcd function
    condition for checking       
    
    check if a is greater than 0"""
    return tf.math.greater(a,0)

def body_gcd(a,b):
    """An auxiliary function for gcd function
    body execution"""
    b = tf.math.floormod(b,a)
    tmp = a; a = b; b = tmp
    return a, b
                          
def gcd(a,b):
    """Greatest Common Divisor in 
    tensorflow implementation
                                  
    GCD(a,b) and returns a single tensor"""
    a = tf.math.abs(a)    #tensorflow absolute value
    b = tf.math.abs(b)
    result = tf.while_loop(condition, body, [a,b])
    return result[1]

"""end of gcd function"""

###################################################

"""start of xgcd function"""

def less_then_0(x, xneg):
    """An auxiliary function for xgcd function
    change the sign of the given parameters

    return a tuple (x, xneg)
    """
    x = -x; xneg = -xneg
    return x, xneg

def equal_0(a, b, a1, b1, a2, b2, true0):
    """An auxiliary function for xgcd function
    
    return a tuple together with the last element for checking
    """
    quot = -tf.math.floordiv(b,a)
    b = tf.math.floormod(b,a)
    a2 = tf.math.add(a2, tf.math.multiply(quot,a1))
    b2 = tf.math.add(b2, tf.math.multiply(quot,b1))
    result = tf.cond(tf.equal(b,0), lambda: (a, b, a1, b1, a2, b2, tf.constant(2)), 
                    lambda: (a, b, a1, b1, a2, b2, tf.constant(0)))
    return result

def while_body_xgcd(a, b, a1, b1, a2, b2, true0):
    """An auxiliary function for xgcd function

    return a tuple together with the last element for checking
    """
    quot = -tf.math.floordiv(a,b)
    a = tf.math.floormod(a,b)
    a1 = tf.math.add(a1, tf.math.multiply(quot,a2))
    b1 = tf.math.add(b1, tf.math.multiply(quot,b2))
    result = tf.cond(tf.equal(a,0), lambda: (a, b, a1, b1, a2, b2, tf.constant(1)), 
                    lambda: equal_0(a, b, a1, b1, a2, b2, true0))
    return result

def cond_xgcd(a, b, a1, b1, a2, b2, true0):
    """An auxiliary function for xgcd function
    
    return true if true0 == 0 otherwise return false
    """
    return tf.equal(true0,0)

def xgcd(a,b):
    a1 = tf.constant(1); b1 = tf.constant(0)
    a2 = tf.constant(0); b2 = tf.constant(1)
    aneg = tf.constant(1); bneg = tf.constant(1)
    a, aneg = tf.cond(tf.less(a,0), lambda: less_then_0(a, aneg), lambda: (a, aneg))
    b, bneg = tf.cond(tf.less(b,0), lambda: less_then_0(b, bneg), lambda: (b, bneg))
    true0 = tf.constant(0)  #true0 works as an indicator for deciding which parameter should be returned
    result = tf.while_loop(cond_xgcd, while_body_xgcd, 
            [a, b, a1, b1, a2, b2, true0])
    a, b, a1, b1, a2, b2, true0 = result
    return tf.cond(tf.equal(true0,1), 
            lambda: (b, tf.math.multiply(a2,aneg), tf.math.multiply(b2,bneg)), 
            lambda: (a, tf.math.multiply(a1,aneg), tf.math.multiply(b1,bneg))) 

"""end of xgcd function"""

#################################################

"""start of inverse_mod function"""

def tru_cond_inverse_mod(xa, n):
    """An auxiliary function for inverse_mod
    
    return a tuple, (the inverse result, and OK message)
    """
    return tf.math.floormod(xa, n), tf.constant("OK")
    
def inverse_mod(a,n):
    """This function computes Compute 1/b mod n.
    
    return a tuple. The second element of the tuple is a string
    that tells whether or not the inverse_mod exists "OK" means
    it exists, otherwise it does not
    """
    (g,xa,xb) = xgcd(a,n)
    result = tf.cond(tf.equal(g,1), lambda: tru_cond_inverse_mod(xa, n), 
            lambda: (tf.constant(0), 
            tf.constant("***** Error *****: value a has no" 
                        "inverse (mod n) as their gcd is g, not 1.")))
    return result
