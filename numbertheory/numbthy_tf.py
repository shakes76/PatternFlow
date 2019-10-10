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
