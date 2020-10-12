"""
numbthy_tf.py contains the following functions

    gcd(a,b)            - Compute the greatest common divisor of a and b.
    xgcd(a,b)           - Find [g,x,y] such that g=gcd(a,b) and g = ax + by.
    power_mod(b,e,n)    - Compute b^e mod n efficiently.
    inverse_mod(b,n)    - Compute 1/b mod n.
    is_prime(n)         - Test whether n is prime using a variety of pseudoprime tests.
    isprimeF(n,b)       - Test whether n is prime or a Fermat pseudoprime to base b.
    isprimeE(n,b)       - Test whether n is prime or an Euler pseudoprime to base b.
    factorone(n)        - Find a factor of n using a variety of methods.
    factorPR(n)         - Find a factor of n using the Pollard Rho method.
"""

import tensorflow as tf

#####################################################################################
##############################  gcd(n)  #############################################
#####################################################################################

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

#####################################################################################
##############################  xgcd(n)  ############################################
#####################################################################################


def less_then_0(x, xneg):
  
  """An auxiliary function for xgcd function
  change the sign of the given parameters
  
  return a tuple (x, xneg)"""
  
  x = -x; xneg = -xneg
  return x, xneg

def equal_0(a, b, a1, b1, a2, b2, true0):
  
  """An auxiliary function for xgcd function
  
  return a tuple togethe with the 
  last element for checking"""
  
  quot = -tf.math.floordiv(b,a)
  b = tf.math.floormod(b,a)
  a2 = tf.math.add(a2, tf.math.multiply(quot,a1))
  b2 = tf.math.add(b2, tf.math.multiply(quot,b1))
  result = tf.cond(tf.equal(b,0), lambda: (a, b, a1, b1, a2, b2, tf.constant(2)), 
                   lambda: (a, b, a1, b1, a2, b2, tf.constant(0)))
  return result

def while_body_xgcd(a, b, a1, b1, a2, b2, true0):
  
  """A auxiliary function for xgcd function
  return a tuple together with the 
  last element for checking"""
  
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
  
  """xgcd takes in two intergers a and b and return a triple (g, x, y) of
  the form g = xa + yb"""
  
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

#####################################################################################
##############################  inverse_mod(n)  #####################################
#####################################################################################


def true_condition_inverse_mod(xa, n):
  
  """An auxiliary function for inverse_mod
  return a tuple, (the inverse result and a constant 1)"""
  
  return tf.math.floormod(xa, n), tf.constant(1)

def inverse_mod(a,n):
  
  """inverse_mod(a,n) - Compute 1/a mod n.
  return a tuple. The second element of the tuple is a string
  that tells whether or not the inverse_mod exists. 1 means it exists,
  otherwise it does not"""
  
  (g,xa,xb) = xgcd(a,n)
  result = tf.cond(tf.equal(g,1), lambda: true_condition_inverse_mod(xa, n), 
                   lambda: (tf.constant(0), 
                            tf.constant(0)))
  return result

#####################################################################################
##############################  power_mod(n)  #######################################
#####################################################################################

def true_condition_0_power_mod(b, e, n):
  
  """An auxiliary function for power_mod"""
  
  e = -e
  b, true0 = inverse_mod(b, n)
  return e, b, true0

def true_condition_1_power_mod(e, i, accum, bpow2, n):
  
  """An auxiliary function for power mod
  This function will be called if conditional_if_power_mod return true"""
  
  accum = tf.math.floormod(tf.math.multiply(accum,bpow2), n)
  return (e, i, accum, bpow2, n)

def while_loop_power_mod(e, i, accum, bpow2, n):
  
  """This function will return the actual result if the inverse_mod is valid
  otherwise this result will never be the return of power_mod"""
  
  e,i,accum,bpow2,n = tf.while_loop(conditional_while_power_mod, 
                         while_body_power_mod, (e,i,accum,bpow2,n))
  return e,i,accum,bpow2,tf.constant(1)

def while_body_power_mod(e, i, accum, bpow2, n):
  
  """A while body for tf.while_loop"""
  
  e, i, accum, bpow2, n = tf.cond(conditional_if_power_mod(e, i, accum, bpow2, n), 
                   lambda:(e, i, accum, bpow2, n),
                   lambda:true_condition_1_power_mod(e, i, accum, bpow2, n))
  bpow2 = tf.math.floormod(tf.math.multiply(bpow2,bpow2), n)
  i = tf.math.add(i,1)
  return e, i, accum, bpow2, n
  
def conditional_while_power_mod(e, i, accum, bpow2, n):
  
  """An auxiliary function for power_mod that return true or false"""
  
  a = tf.bitwise.right_shift(e, i)
  return tf.greater(a,0)

def conditional_if_power_mod(e, i, accum, bpow2, n):
  
  """An auxiliary function for power_mod that return true of false"""
  
  a = tf.bitwise.right_shift(e,i)
  a = tf.bitwise.bitwise_and(a,1)
  return tf.equal(a,0)


def power_mod(b,e,n):
  
  """power_mod(b,e,n) computes the eth power of b mod n.
  (Actually, this is not needed, as pow(b,e,n) does the same thing for positive integers.
  This will be useful in future for non-integers or inverses.)"""
  
  #true0 is a parameter to check if there is an inverse_mod, if there is true0 will be 1
  e, b, true0 = tf.cond(tf.less(e,0), lambda:true_condition_0_power_mod(b,e,n), 
                        lambda:(e, b, tf.constant(1)))  # Negative powers can be computed if gcd(b,n)=1
  accum = tf.constant(1); i = tf.constant(0); bpow2 = b
  #check if there is an inverse_mod
  e, i, accum, bpow2, n = tf.cond(tf.equal(true0, 0), 
                                  lambda:(-1,-1,-1,0,0), 
                                  lambda:while_loop_power_mod(e,i,accum,bpow2,n))
	
  return accum, n

#####################################################################################
##############################  isprimeF(n)  ########################################
#####################################################################################

def isprimeF(n, b):
  
  """return true if n is a prime or a Fermat pseudoprime to base b
  return false otherwise"""
  
  m = power_mod(b, n-1, n)
  return tf.equal(m[0], 1)

#####################################################################################
##############################  isprimeE(n)  ########################################
#####################################################################################

def true_condition_3_isprimeE(n, b):
  
  """An auxiliary function for isprimeE that returns true or false"""
  
  r = tf.math.add(n,-1)
  r = tf.while_loop(while_0_isprimeE, lambda r: tf.math.floordiv(r,2), [r])
  m = tf.math.pow(b, r)
  c = tf.math.floormod(m, n)
  result = tf.cond(tf.equal(c,1), 
          lambda: tf.equal(1,1), 
          lambda: false_condition_3_isprimeE(n, c))
  return result

def false_condition_3_isprimeE(n, c):
  
  """An auxiliary function for isprimeE that returns true of false"""
  
  #true0 is a parameter that checks for conditions
  true0 = tf.constant(0)
  n, c, true0 = tf.while_loop(lambda n, c, true0: tf.equal(true0,0), 
                              while_1_isprimeE, [n, c, true0])
  result = tf.cond(tf.equal(true0, 2), 
                   lambda: tf.equal(1,1), 
                   lambda: tf.equal(0,1))
  return result

def false_condition_2_isprimeE(n, c, true0):
  
  """An auxiliary function for isprimeE"""
  
  m = tf.math.pow(c, 2)
  c = tf.math.floormod(m, n)
  return n, c, tf.constant(0)

def false_condition_1_isprimeE(n, c, true0):
  
  """An auxiliary function for isprimeE"""
  
  n, c, true0 = tf.cond(tf.equal(c, n-1), 
                  lambda: (n, c, tf.constant(2)), 
                  lambda: false_condition_2_isprimeE(n, c, true0))
  return n, c, true0 

def while_0_isprimeE(r):
  
  """An auxiliary function for isprimeE"""
  
  a = tf.math.floormod(r,2)
  return tf.equal(a,0)

def while_1_isprimeE(n, c, true0):
  
  """An auxiliary function for isprimeE"""
  
  n, c, true0 = tf.cond(tf.equal(c,1), 
                  lambda: (n, c, tf.constant(1)), 
                  lambda: false_condition_1_isprimeE(n, c, true0))
  return n, c, true0

def isprimeE(n, b):
  
	"""isprimeE(n) - Test whether n is prime or an Euler pseudoprime to base b."""
  
	result = tf.cond(isprimeF(n,b), 
                   lambda: true_condition_3_isprimeE(n, b), 
                   lambda: tf.equal(0,1))
	return result

#####################################################################################
##############################  is_prime(n)  ########################################
#####################################################################################

def conditional_2_is_prime(n):
  
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

def conditional_1_is_prime(n):
  
  """An auxiliary function that return true if n is
  one of the obvious prime between 2 and 29 if not 
  evaluate isprimeE against 2, 3 and 5"""
  
  result = tf.cond(conditional_2_is_prime(n), 
                   lambda: tf.equal(1,1), 
                   lambda: conditional_0_is_prime(n))
  return result 

def conditional_0_is_prime(n):
  
  """An auxiliary function that return true if n is psudo Euler prime
  against 2, 3 and 5"""
  
  return tf.math.logical_and(tf.math.logical_and(isprimeE(n,2), 
                                                 isprimeE(n,3)), isprimeE(n,5))
 
def is_prime(n):
  
  """Evaluate if n is prime or not"""
  
  n = tf.cond(tf.less(n,0), lambda: -n, lambda: n)
  result = tf.cond(tf.less(n,2), lambda: tf.equal(0,1), 
                   lambda: conditional_1_is_prime(n))
  return result

#####################################################################################
##############################  factorPR(n)  ########################################
#####################################################################################

def conditional_0_factorPR(n, true0, g):
  
  """Au auxiliary function: if g == n, then true0 will be assigned
  by 0 and then the function break and return 1 otherwise it breaks and
  return g"""
  
  result = tf.cond(tf.equal(g,n), 
                   lambda:(n, tf.constant(0), g), 
                   lambda:(n, tf.constant(1), g))
  return result
              
def while_body_0_factorPR(additive, slow, fast, numsteps, i, n, true0, g):
  
  """An auxiliary function that executes one of the while bodies"""
  
  slow = tf.math.floormod(tf.math.add(tf.math.multiply(slow,slow),additive),n)
  i = tf.math.add(i,1)
  fast = tf.math.floormod(tf.math.add(tf.math.multiply(fast,fast),additive),n)
  fast = tf.math.floormod(tf.math.add(tf.math.multiply(fast,fast),additive),n)
  g = gcd(tf.math.subtract(fast,slow),n)
  n, true0, g = tf.cond(tf.equal(g,1), 
                        lambda:(n, true0, g), 
                        lambda:conditional_0_factorPR(n, true0, g))
  return additive, slow, fast, numsteps, i, n, true0, g
  
def while_conditional_0_factorPR(additive, slow, fast, numsteps, i, n, true0, g):
  
  """An auxiliary function to determine whether or not the while 
  loop should break based on the true0 parameter"""
  
  return tf.cond(tf.equal(true0,3), 
                 lambda:tf.less(i,numsteps), 
                 lambda:tf.equal(0,1))

def while_body_1_factorPR(numsteps, additive, n, true0, g):
  
  """An auxiliary function to execute one of the while bodies"""
  
  fast = tf.constant(1)
  slow = tf.constant(1)
  i = tf.constant(1)
  additive, slow, fast, numsteps, i, n, true0, g = tf.while_loop(while_conditional_0_factorPR, 
                         while_body_0_factorPR, 
                         [additive, slow, fast, numsteps, i, n, true0, g])
  additive = tf.math.add(additive,1)
  return numsteps, additive, n, true0, g

def while_conditional_1_factorPR(numsteps, additive, n, true0, g):
  
  """An auxiliary function to determine whether the while loop should break
  based on the true0 parameter"""
  
  return tf.cond(tf.equal(true0,3), 
                 lambda:tf.less(additive,5), 
                 lambda:tf.equal(0,1))
  
def factorPR(n):
  
  """Using Pollard Pho method to find a factor of n"""
  
  numsteps = tf.math.multiply(tf.constant(2.0),
                              tf.math.floor(tf.math.sqrt(tf.math.sqrt(
                                            (tf.dtypes.cast
                                             (n,tf.float32))))))
  numsteps = tf.dtypes.cast(numsteps,tf.int32)                            
  additive = tf.constant(1)
  
  #true0 is a parameter to determine the return should be g or 1
  true0 = tf.constant(3)
  g = tf.constant(0)
  result = tf.while_loop(while_conditional_1_factorPR, 
                         while_body_1_factorPR, 
                         [numsteps, additive, n, true0, g])
  return tf.cond(tf.equal(result[3],1), 
                 lambda:result[4], 
                 lambda:tf.constant(1))

#####################################################################################
##############################  factorone(n)  #######################################
#####################################################################################

def conditional_0_if_factorone(n, fact, check, true0):
  
  """An auxiliary function for factorone that assigns true0 and check
  to certain values"""
  
  a = tf.math.floormod(n,fact)
  result = tf.cond(tf.equal(a,0), 
                     lambda:(n, fact, tf.constant(1), tf.constant(0)), 
                     lambda:(n, fact, check, tf.constant(1)))
  return result

def for_loop_factorone(n, fact, check, true0):
  
  """An auxiliary function that loops over conditional_0_if_factorone"""
  
  result = tf.cond(tf.equal(true0,0), 
                   lambda:(n,fact,check,true0), 
                   lambda:conditional_0_if_factorone(n,fact,check,true0))
  return result
    
def conditional_1_if_factorone(n, fact, true0):
  
  """An auxiliary function that determine which result should return"""
  
  result = tf.cond(tf.equal(true0,0), lambda:fact, lambda:factorPR(n))
  return result
  
def factorone(n):
  
  """Find a prime factor of n using variety of methods"""
  
  true0 = tf.constant(2)
  check = tf.constant(0)
  fact = tf.constant(0)
  for fact1 in (2,3,5,7,11,13,17,19,23,29):
    fact1 = tf.constant(fact1)
    fact = tf.cond(tf.equal(check,1), lambda:fact, lambda:fact1)
    n, fact, check, true0 = tf.cond(is_prime(n), 
                                    lambda:(n,fact,check,true0), 
                                    lambda:for_loop_factorone(n,fact,check,true0))
  result = tf.cond(tf.equal(true0,2), 
                   lambda:n, 
                   lambda:conditional_1_if_factorone(n,fact, true0))
  return result
