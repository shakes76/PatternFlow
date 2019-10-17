import tensorflow as tf

def cond(a, b, quotients):
    # Any oter variable but a is not used here
    # They are only parsed because of structure of tf.while_loop
    return tf.greater(a, tf.constant(0))
    
def body(a, b, quotients):
    quotient = tf.floor_div(b, a)
    _temp = a
    a = tf.mod(b, a)
    b = _temp

    
    x0 = quotients[0]
    y0 = quotients[1]
    x1 = quotients[2]
    y1 = quotients[3]    
    
    _temp = x1
    x1 = tf.subtract(x0, tf.multiply(quotient, x1))
    x0 = _temp
    
    _temp = y1
    y1 = tf.subtract(y0, tf.multiply(quotient, y1))
    y0 = _temp
    
    quotients = (x0, y0, x1, y1)
    
    return a, b, quotients


def extended_gcd(a, b):
    '''
    Return the Greatest Common Divisior of two interger values (a and b)
    '''
    # Only whole numbers 
    a = abs(a)
    b = abs(b)
    
    ### Initialize tensors ###
    a = tf.Variable(a)
    b = tf.Variable(b)
    
    # Quotients 
    x0 = tf.Variable(0)
    y0 = tf.Variable(1)
    x1 = tf.Variable(1)
    y1 = tf.Variable(0)

    
    # While a
    loop = tf.while_loop(cond, body, [a, b, (x0, y0, x1, y1)])
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(loop)
        print(result)
        gcd = result[1]
        
        return gcd