import tensorflow as tf


def gamma_correction(input_img, gamma_coef=1.0):
    '''
    Apply gamma correction to input.
    :param input: A ndarray contains the information of single image, which can be rgb, rgba or grey.
    :param gamma_coef: The gamma correction coef, default set to 1.0.
    :return: The ndarray contains the image after gamma correction in uin8 format.
    '''

    # Convert input numpy array into tensorflow constant.
    input_tensor = tf.constant(input_img, tf.float32)
    # Do normalization for input image. 0~255 to 0~1.
    norm = tf.div(input_tensor, 256)
    # Apply gamma correction.
    adjust = tf.math.pow(norm, gamma_coef)
    # Re-normalization for result image. 0~1 to 0~255.
    inv_norm = tf.multiply(adjust, 256)
    # Convert from float32 to uint8.
    to_uint8 = tf.cast(inv_norm, tf.uint8)
    # Rename
    gamma = to_uint8

    # Initialization.
    init = tf.compat.v1.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        # Run gamma correction.
        result = sess.run(gamma)
        sess.close()

        # Return result.
        return result
