import tensorflow as tf

def cond(a, b, quotients):
    '''
    Condition of while loop
    
    Will check if a is greater than 0
    
    Args: 
        param 1 (Tensor int):        First value, a
        param 2 (Tensor int):        Second value, b
        param 3 (list<Tensor int>):  Quotient values,
        
    Returns:
        Tensor bool:  True for succes, False otherwise
    '''
    # Any oter variable but a is not used here...
    # They are only parsed because of the structure of tf.while_loop requires so
    
    return tf.greater(a, tf.constant(0))
    
def body(a, b, quotients):
    '''
    Body of while loop
    
    Will update values of a, b and quotients
    
    Args: 
        param 1 (Tensor int):        First value, a
        param 2 (Tensor int):        Second value, b
        param 3 (list<Tensor int>):  Quotient values,
        
    Returns:
        Tensor int:        Updated value of a
        Tensor int:        Updated value of b
        list<Tensor int>:  Updated values of Quotients
    '''
    
    quotient = tf.floor_div(b, a)
    
    # Update values of a and b
    _temp = a
    a = tf.mod(b, a)
    b = _temp

    x0 = quotients[0]
    y0 = quotients[1]
    x1 = quotients[2]
    y1 = quotients[3]    
    
    # Update values of x quotient
    _temp = x1
    x1 = tf.subtract(x0, tf.multiply(quotient, x1))
    x0 = _temp
    
    # Update values of y quotient
    _temp = y1
    y1 = tf.subtract(y0, tf.multiply(quotient, y1))
    y0 = _temp
    
    quotients = (x0, y0, x1, y1)
    
    return a, b, quotients


def extended_gcd(a, b):
    '''
    Determine the Greatest Common Divisior of two interger values (a and b)
    as well as the quotients needed to satisfy ax + by = gcd(a, b)
    
    Args: 
        param 1 (Tensor int):        First value, a
        param 2 (Tensor int):        Second value, b
        param 3 (list<Tensor int>):  Quotient values,
        
    Returns:
        int:        The value of the Greatest Common Divisior of a and b
        list<int>:  The values of the quotients that satisfies ax + by = gcd(a, b)
    '''
    
    # Only whole numbers 
    a = abs(a)
    b = abs(b)
    
    # Initialize tensors
    a = tf.Variable(a)
    b = tf.Variable(b)
    
    # Quotients 
    x0 = tf.Variable(0)
    y0 = tf.Variable(1)
    x1 = tf.Variable(1)
    y1 = tf.Variable(0)

    
    # While a is greater than 0 find gcd(a, b) as well as quotients
    loop = tf.while_loop(cond, body, [a, b, (x0, y0, x1, y1)])
    
    # Initialize tensor session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(loop)
        gcd = result[1]
        quotients = (result[2][0],result[2][1])
        
        return gcd, quotients