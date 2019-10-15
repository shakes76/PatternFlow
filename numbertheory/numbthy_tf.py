
def condition_gcd(a,b):
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
    result = tf.while_loop(condition_gcd, body_gcd, [a,b])
    return result[1]

def less_then_0(x, xneg):
    """An auxiliary function for xgcd function
    change the sign of the given parameters
  
    return a tuple (x, xneg)"""
    x = -x; xneg = -xneg
    return x, xneg

def equal_0(a, b, a1, b1, a2, b2, true0):
    """An auxiliary function for xgcd function
    return a tuple togethe with the last element for checking"""
    quot = -tf.math.floordiv(b,a)
    b = tf.math.floormod(b,a)
    a2 = tf.math.add(a2, tf.math.multiply(quot,a1))
    b2 = tf.math.add(b2, tf.math.multiply(quot,b1))
    result = tf.cond(tf.equal(b,0), lambda: (a, b, a1, b1, a2, b2, tf.constant(2)), 
                   lambda: (a, b, a1, b1, a2, b2, tf.constant(0)))
    return result

def while_body_xgcd(a, b, a1, b1, a2, b2, true0):
    """A auxiliary function for xgcd function
    return a tuple together with the last element for checking"""
    quot = -tf.math.floordiv(a,b)
    a = tf.math.floormod(a,b)
    a1 = tf.math.add(a1, tf.math.multiply(quot,a2))
    b1 = tf.math.add(b1, tf.math.multiply(quot,b2))
    result = tf.cond(tf.equal(a,0), lambda: (a, b, a1, b1, a2, b2, tf.constant(1)), 
                  lambda: equal_0(a, b, a1, b1, a2, b2, true0))
    return result

def cond_xgcd(a, b, a1, b1, a2, b2, true0):
    """An auxiliary function for xgcd function
    return true if true0 == 0 otherwise return false"""
    return tf.equal(true0,0)

def xgcd(a,b):
    a1 = tf.constant(1); b1 = tf.constant(0)
    a2 = tf.constant(0); b2 = tf.constant(1)
    aneg = tf.constant(1); bneg = tf.constant(1)
    a, aneg = tf.cond(tf.less(a,0), lambda: less_then_0(a, aneg), lambda: (a, aneg))
    b, bneg = tf.cond(tf.less(b,0), lambda: less_then_0(b, bneg), lambda: (b, bneg))
    true0 = tf.constant(0)
    result = tf.while_loop(cond_xgcd, while_body_xgcd, 
                        [a, b, a1, b1, a2, b2, true0])
    a, b, a1, b1, a2, b2, true0 = result
    return tf.cond(tf.equal(true0,1), 
                   lambda: (b, tf.math.multiply(a2,aneg), tf.math.multiply(b2,bneg)), 
                   lambda: (a, tf.math.multiply(a1,aneg), tf.math.multiply(b1,bneg)))

def tru_cond_inverse_mod(xa, n):
    """An auxiliary function for inverse_mod
    return a tuple, (the inverse result and OK message)"""
    return tf.math.floormod(xa, n), tf.constant("OK")
  
def inverse_mod(a,n):
    """inverse_mod(b,n) - Compute 1/b mod n.
    return a tuple. The second element of the tuple is a string
    that tells whether or not the inverse_mod exists. OK means it exists,
    otherwise it does not"""
    (g,xa,xb) = xgcd(a,n)
    result = tf.cond(tf.equal(g,1), lambda: tru_cond_inverse_mod(xa, n), 
                   lambda: (tf.constant(0), 
                            tf.constant("***** Error *****: value a has no" 
                            "inverse (mod n) as their gcd is g, not 1.")))
    return result

"""The following function has problem"""

def tru_cond0_power_mod(b, e, n):
    e = -e
    b = inverse_mod(b, n)
    return e, b

def cond_while(e, i):
    a = tf.bitwise.right_shift(e, i)
    return tf.greater(a,0)

def cond_if(e, i, accum, bpow2, n):
    a = tf.bitwise.right_shift(e, i)
    return tf.bitwise.bitwise_and(a, 1)

def tru_cond1_power_mod(e, i, accum, bpow2, n):
    accum = tf.math.floormod(tf.math.multiply(accum*bpow2), n)
    bpow2 = tf.math.floormod(tf.math.multiply(bpow2*bpow2), n)
    i.assign_add(1)
    return e, i, accum, bpow2, n

def fal_cond1_power_mod(e, i, accum, bpow2, n):
    bpow2 = tf.math.floormod(tf.math.multiply(bpow2*bpow2), n)
    i.assign_add(1)
    return e, i, accum, bpow2, n

def power_mod(b,e,n):
    """power_mod(b,e,n) computes the eth power of b mod n.
    (Actually, this is not needed, as pow(b,e,n) does the same thing for positive integers.
    This will be useful in future for non-integers or inverses.)"""
    e, b = tf.cond(tf.less(e,0), lambda: tru_cond0_power_mod(b,e,n), lambda: (e, b))  # Negative powers can be computed if gcd(b,n)=1
    accum = tf.constant(1); i = tf.constant(0); bpow2 = b
    e, i, accum, bpow2, n = tf.while_loop(cond_while, tru_cond1_power_mod, 
                                       fal_cond1_power_mod, 
                                       [e, i, accum, bpow2, n])
	return accum

"""###############################################"""

def isprimeF(n, b):
    """small numbers can be used
    but if n and b get too large value
    overflow may occur
    the author tried n = 31, b = 3 and
    it doesn't work as intended"""
    m = tf.math.pow(b, n-1)
    result = tf.math.floormod(m, n)
    return tf.equal(result, 1)

def fal2_cond_isprimeE(n, c, true0):
    m = tf.math.pow(c, 2)
    c = tf.math.floormod(m, n)
    return n, c, tf.constant(0)

def fal1_cond_isprimeE(n, c, true0):
    n, c, true0 = tf.cond(tf.equal(c, n-1), 
                  lambda: (n, c, tf.constant(2)), 
                  lambda: fal2_cond_isprimeE(n, c, true0))
    return n, c, true0 

def while0_isprimeE(r):
    a = tf.math.floormod(r,2)
    return tf.equal(a,0)

def while1_isprimeE(n, c, true0):
    n, c, true0 = tf.cond(tf.equal(c,1), 
                  lambda: (n, c, tf.constant(1)), 
                  lambda: fal1_cond_isprimeE(n, c, true0))
    return n, c, true0

def fal_cond_isprimeE(n, c):
    true0 = tf.constant(0)
  
    n, c, true0 = tf.while_loop(lambda n, c, true0: tf.equal(true0,0), 
                              while1_isprimeE, [n, c, true0])
  
    result = tf.cond(tf.equal(true0, 2), 
                   lambda: tf.equal(1,1), 
                   lambda: tf.equal(0,1))
    return result

def tru_cond_isprimeE(n, b):
    r = tf.math.add(n,-1)
    r = tf.while_loop(while0_isprimeE, lambda r: tf.math.floordiv(r,2), [r])
    m = tf.math.pow(b, r)
    c = tf.math.floormod(m, n)
    result = tf.cond(tf.equal(c,1), lambda: tf.equal(1,1), lambda: fal_cond_isprimeE(n, c))
    return result

def isprimeE(n, b):
    """isprimeE(n) - Test whether n is prime or an Euler pseudoprime to base b."""
    result = tf.cond(isprimeF(n,b), lambda: tru_cond_isprimeE(n, b), lambda: tf.equal(0,1))
    return result

def cond2_is_prime(n):
    """An auxiliary function that check 
    if n in [2,3,5,7,11,13,17,19,23,29]
    return true if n is one of those false otherwise"""
    compare0 = tf.math.logical_or(tf.equal(n,2), tf.equal(n,3))
    compare1 = tf.math.logical_or(tf.equal(n,5), tf.equal(n,7))
    compare2 = tf.math.logical_or(tf.equal(n,11), tf.equal(n,13))
    compare3 = tf.math.logical_or(tf.equal(n,17), tf.equal(n,19))
    compare4 = tf.math.logical_or(tf.equal(n,23), tf.equal(n,29))
    compare5 = tf.math.logical_or(compare0, compare1)
    compare6 = tf.math.logical_or(compare2, compare3)
    compare7 = tf.math.logical_or(compare4, compare5)
    compare8 = tf.math.logical_or(compare6, compare7)
    return compare8

def cond1_is_prime(n):
    """An auxiliary function that return true if n is
    one of the obvious prime between 2 and 29 if not 
    evaluate isprimeE against 2, 3 and 5"""
    result = tf.cond(cond2_is_prime(n), lambda: tf.equal(1,1), lambda: cond0_is_prime(n))
    return result 

def cond0_is_prime(n):
    """An auxiliary function that return true if n is psudo Euler prime
    against 2, 3 and 5"""
    return tf.math.logical_and(tf.math.logical_and(isprimeE(n,2), isprimeE(n,3)),
																						  isprimeE(n,5))
 
def is_prime(n):
    """Evaluate if n is prime or not"""
    n = tf.cond(tf.less(n,0), lambda: -n, lambda: n)
    result = tf.cond(tf.less(n,2), lambda: tf.equal(0,1), lambda: cond1_is_prime(n))
    return result



def factorPR(n):
    """factorPR(n) - Find a factor of n using the Pollard Rho method.
    Note: This method will occasionally fail."""
    numsteps=2*math.floor(math.sqrt(math.sqrt(n)))
    for additive in range(1,5):
		fast=slow=1; i=1
		while i<numsteps:
			slow = (slow*slow + additive) % n
			i = i + 1
			fast = (fast*fast + additive) % n
			fast = (fast*fast + additive) % n
			g = gcd(fast-slow,n)
			if (g != 1):
				if (g == n):
					break
				else:
					return g
	return 1
