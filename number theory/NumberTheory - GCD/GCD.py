import tensorflow as tf

class GCD:
    
    def __init__(self, a, b):
        self.a = tf.Variable(a)
        self.b = tf.Variable(b)
        
        loop = tf.while_loop(self.cond, self.body, [self.a, self.b])   
        
        with tf.Session() as sess:
            result = sess.run(loop)
            gcd = result[1]
            return gcd
        
    def cond(self, a, b):
        return tf.greater(a, tf.constant(0))
    
    def body(self, a, b):
        b = tf.mod(b, a)
        temp = a
        a = b
        b = temp
        return a, b