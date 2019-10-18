import tensorflow as tf

class GCDExtended:
    '''
    Find the Greatest Common Divisior of two interger values (a and b)
    as well as the quotiens needed to satisfy ax + by = gcd(a, b)
    '''
    
    def __init__(self, a, b):
        self.a = tf.Variable(a)
        self.b = tf.Variable(b)
        self.gcd = None
        self.x = 0
        self.y = 1
        self.x1 = 1
        self.y1 = 0
        
        # While a is greater than 0 find gcd(a, b) as well as quotients
        loop = tf.while_loop(self.cond, self.body, [self.a, self.b, (self.x, self.y, self.x1, self.y1)])  
        
        # Initialize tensor session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(loop)
            self.gcd = result[1]
            self.x = result[2][0]
            self.y = result[2][1]
            
            print('Greatest Common Divisior of', a, ' and ', b, ' is: ', self.gcd)
            print('Quotients for the gcd are (', self.x, ',', self.y, ')')
        
    def cond(self, a, b, quotients):
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
        # Any oter variable but a is not used here
        # They are only parsed because of structure of tf.while_loop
        
        return tf.greater(a, tf.constant(0))
    
    def body(self, a, b, quotients):
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
    
