import tensorflow as tf
import math
import functools

def condition(a,b):
    """An auxiliary function for gcd function
    condition for checking       
    
    check if a is greater than 0"""
    return tf.math.greater(a,0)

def body(a,b):
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

def xgcd(a,b):
    """Take two integers a and b 
    then evaluate for the equation 
    that satisfies g = ax + by"""
    a1 = tf.constant(1); b1 = tf.constant(0)
    a2 = tf.constant(0); b2 = tf.constant(1)
    aneg = tf.constant(1); bneg = tf.constant(1)
    if(a.eval() < 0):
        a = -a; aneg = -aneg
    if(b.eval() < 0):
        b = -b; bneg = -bneg
    while(1):
        quot = -tf.math.floordiv(a,b)
        a = tf.math.floormod(a,b)
        a1 = tf.math.add(a1, tf.math.multiply(quot,a2))
        b1 = tf.math.add(b1, tf.math.multiply(quot,b2))
        if(a.eval() == 0):
            return (b, tf.math.multiply(a2,aneg), tf.math.multiply(b2,bneg))
        quot = -tf.math.floordiv(b,a)
        b = tf.math.floormod(b,a)
        a2 = tf.math.add(a2, tf.math.multiply(quot,a1))
        b2 = tf.math.add(b2, tf.math.multiply(quot,b1)) 
        if(b.eval() == 0):
            return (a, tf.math.multiply(a1,aneg), tf.math.multiply(b1,bneg))
