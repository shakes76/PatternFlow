import tensorflow as tf

class GCD:
    
    def __init__(self, a, b):
        self.a = tf.Variable(a)
        self.b = tf.Variable(b)
        self.gcd = None
        
        # While a is greater than 0 find gcd(a, b)
        loop = tf.while_loop(self.cond, self.body, [self.a, self.b])  
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result = sess.run(loop)
            self.gcd = result[1]
            
            print('Greatest Common Divisior of', a, ' and ', b, ' is: ', self.gcd)
        
    def cond(self, a, b):
        return tf.greater(a, tf.constant(0))
    
    def body(self, a, b):
        b = tf.mod(b, a)
        temp = a
        a = b
        b = temp
        return a, b
    
