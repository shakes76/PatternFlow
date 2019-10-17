import tensorflow as tf

def adjust_sigmoid(image, cutoff=0.5, gain=10.0, inv=False):
    '''
    Applies sigmoid correction (also known as contrast adjustment) to the input image.
    '''



    input_image = tf.constant(image)

    limits = input_image.dtype.limits
    normalizer = limits[1] - limits[0] 
    normalized = tf.divide(image, normalizer)
    
    cut_off = tf.math.subtract(tf.constant(cutoff, tf.float32), normalized)
    gained = tf.multiply(tf.constant(gain, tf.float32), cut_off)
    exp = tf.math.exp(gained)
    add_one = tf.math.add(exp, 1.0)
    divider = tf.math.divide(1.0, add_one)
    condition = tf.cond(tf.cast(inv, tf.bool), lambda: tf.subtract(1.0, divider), lambda: divider)
    output = tf.multiply(condition, normalizer)

    init = tf.compat.v1.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        adjusted_image = sess.run(output)
        sess.close()
        return adjusted_image



