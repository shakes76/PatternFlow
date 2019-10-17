import tensorflow as tf


def cond(a, b):
    # b is not used here 
    # They are only parsed because of structure of tf.while_loop
    
    return tf.greater(a, tf.constant(0))
    
def body(a, b):
    b = tf.mod(b, a)
    temp = a
    a = b
    b = temp
    
    return a, b

def gcd(a, b):
    '''
    Return the Greatest Common Divisior of two interger values (a and b)
    '''
    # Only whole numbers 
    a = abs(a)
    b = abs(b)
    
    ### Initialize tensors ###
    a = tf.Variable(a)
    b = tf.Variable(b)
    
    # While a is greater than 0 find gcd(a, b)
    loop = tf.while_loop(cond, body, [a, b])  
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(loop)
        gcd = result[1]
        
        return gcd