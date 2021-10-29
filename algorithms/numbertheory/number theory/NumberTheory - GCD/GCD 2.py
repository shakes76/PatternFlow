import tensorflow as tf

class GCD:
    
    def __init__(self, a, b):
        self.a = tf.Variable(a)
        self.b = tf.Variable(b)
        self.gcd = None
        
        # While a is greater than 0 find gcd(a, b)
        loop = tf.while_loop(self.cond, self.body, [self.a, self.b])  
        
        # Initialize tensor session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(loop)
            self.gcd = result[1]
        
    def cond(self, a, b):
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
    
    def body(self, a, b):
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
    
