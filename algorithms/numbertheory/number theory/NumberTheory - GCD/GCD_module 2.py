import tensorflow as tf


def cond(a, b):
    '''
    Condition of while loop
    
    Will check if a is greater than 0
    
    Args: 
        param 1 (Tensor int):        First value, a
        param 2 (Tensor int):        Second value, b
        
    Returns:
        Tensor bool:  True for succes, False otherwise
    '''
    # b is not used here 
    # They are only parsed because of structure of tf.while_loop
    
    return tf.greater(a, tf.constant(0))
    


def body(a, b):
    '''
    Body of while loop
    
    Will update values of a and b
    
    Args: 
        param 1 (Tensor int):  First value, a
        param 2 (Tensor int):  Second value, b
        
    Returns:
        Tensor int:  Updated value of a
        Tensor int:  Updated value of b
    '''
    
    # Update values of y quotient
    b = tf.mod(b, a)
    temp = a
    a = b
    b = temp
    
    return a, b



def gcd(a, b):
    '''
    Determine the Greatest Common Divisior of two interger values (a and b)
    
    Args: 
        param 1 (Tensor int):  First value, a
        param 2 (Tensor int):  Second value, b
        
    Returns:
        int:  The value of the Greatest Common Divisior of a and b
    '''
    
    # Only whole numbers 
    a = abs(a)
    b = abs(b)
    
    # Initialize tensors
    a = tf.Variable(a)
    b = tf.Variable(b)
    
    # While a is greater than 0 find gcd(a, b)
    loop = tf.while_loop(cond, body, [a, b])  
        
    # Initialize tensor session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(loop)
        gcd = result[1]
        
        return gcd