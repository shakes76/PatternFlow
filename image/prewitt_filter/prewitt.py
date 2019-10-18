import tensorflow as tf

def prewitt_filter(image):
    #convolution for vertical gradient approximation
    vertical = tf.constant([[1,0,-1],[1,0,-1],[1,0,-1]],dtype=tf.float32)
    vertical = tf.reshape(vertical,[3,3,1,1])

    #convolution for horizontal gradient approximation
    horizontal = tf.constant([[1,1,1],[0,0,0],[-1,-1,-1]],dtype=tf.float32)
    horizontal = tf.reshape(horizontal,[3,3,1,1])

    expand_img = tf.reshape(image,[1,tf.shape(image)[0],tf.shape(image)[1],1])
    v_grad = tf.nn.conv2d(expand_img,vertical, strides=[1,1,1,1], padding="SAME")
    h_grad = tf.nn.conv2d(expand_img,horizontal, strides=[1,1,1,1], padding="SAME")
    img = tf.math.sqrt(tf.math.square(v_grad) + tf.math.square(h_grad))
    return img

