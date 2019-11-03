import tensorflow as tf

def adjust_sigmoid(image, cutoff=0.5, gain=10.0, inv=False):
    '''
    Applies sigmoid correction (also known as contrast adjustment) to the input image.

    Parameters:

        image: ndarray
            Input image to which sigmoid correction is to be applied.

        cutoff: float
            Adjusts the horizontal shift of the sigmoid curve.
            Defaults to 0.5.

        gain: float
            Adjusts the slope of the sigmoid curve:
            Defaults to 10.0

        inv: bool
            If True, the negative sigmoid correction is used.
            Defaults to False

    Returns:

        adjusted_image: ndarray
            The resultant image when sigmoid correction is applied to the input image.
    '''

    # Convert the image to tf constant
    input_image = tf.constant(image)

    # Determine limits
    # Note that if the dtype of the pixels in image are floats, they
    # should have values between 0 and 1
    limits = input_image.dtype.limits
    normalizer = limits[1] - limits[0]

    # Normalize the inputs
    normalized = tf.divide(image, normalizer)

    # Cutoff for horizontal shift
    cut_off = tf.math.subtract(tf.constant(cutoff, tf.float32), normalized)

    # Gain for slope of sigmoid function
    gained = tf.multiply(tf.constant(gain, tf.float32), cut_off)
    exp = tf.math.exp(gained)

    # Perform 1 / (1 + exp)
    add_one = tf.math.add(exp, 1.0)
    divider = tf.math.divide(1.0, add_one)

    # 1 - divider if inv==True
    condition = tf.cond(tf.cast(inv, tf.bool), lambda: tf.subtract(1.0, divider), lambda: divider)

    # Revert the initial normalization
    output = tf.multiply(condition, normalizer)

    # Initialize variables and run the session.
    init = tf.compat.v1.global_variables_initializer()
    
    with tf.Session() as sess:

        # Run session with output
        sess.run(init)
        adjusted_image = sess.run(output)
        sess.close()

        # Return adjusted image
        return adjusted_image
