# The implement of sigmoid Correction according to the original adjust_sigmoid function
# =======================
import tensorflow as tf

def sigmoid(input_img, cutoff=0.5, gain=10, inv=False):
    """
    Parameters
        image(ndarray):
            Input image.

        cutoff(float, optional):
            Cutoff of the sigmoid function that shifts the characteristic curve
            in horizontal direction. Default value is 0.5.

        gain(float, optional):
            The constant multiplier in exponential’s power of sigmoid function. Default value is 10.

        inv(bool, optional):
            If True, returns the negative sigmoid correction. Defaults to False.

    Returns:
        out(ndarray):
            Sigmoid corrected output image.
    """
        
    # Transform the input_img to a tensor
    input_tensor = tf.constant(input_img, tf.float32)

    normalize = input_tensor / 255
    #calculate (equation : O = 1/(1 + exp*(gain*(cutoff - I))))
    if inv:
      sigmoid = 1.0 - 1.0 / (1.0 + tf.math.exp(gain * (cutoff - input_tensor)))
    else:
      sigmoid = 1.0 / (1.0 + tf.math.exp(gain * (cutoff - input_tensor)))
      
    re_normalize = sigmoid * 255

    init = tf.compat.v1.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        out = tf.to_int32(re_normalize)
        result = sess.run(out)
        return result
