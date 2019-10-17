import tensorflow as tf

def sigmoid(input_img, cutoff=0.5, gain=10, inv=False):
    
    # Transform the input_img to a tensor
    input_tensor = tf.constant(input_img, tf.float32)

    normalize = input_tensor / 256
    #calculate (equation : O = 1/(1 + exp*(gain*(cutoff - I))))
    if inv:
      sigmoid = 1.0 - 1.0 / (1.0 + tf.math.exp(gain * (cutoff - input_tensor)))
    else:
      sigmoid = 1.0 / (1.0 + tf.math.exp(gain * (cutoff - input_tensor)))
      
    re_normalize = sigmoid * 256

    init = tf.compat.v1.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        out = tf.to_int32(re_normalize)
        result = sess.run(out)
        return result
