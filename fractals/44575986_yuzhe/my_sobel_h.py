import tensorflow as tf

def tf_sobel_h(image):

    """Find the vertical edges of an image using the Sobel transform.
    Parameters
    ----------
    image : 2-D array
        Image to process
    Returns
    -------
    output : 2-D array
        The Sobel edge map.
    Notes
    -----
    We use the following kernel::
        1   0  -1
        2   0  -2
        1   0  -1
    """
    image = tf.Variable(image, dtype=tf.float64)
    if len(image.get_shape().as_list()) != 2:
        raise ValueError("the image dimension is not 2")
    # expand both image and weight dimension to 4d
    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, 3)
    HSOBEL_WEIGHTS = tf.Variable([[-1, -2, -1],
                                   [0, 0, 0],
                                    [1, 2, 1]], dtype=tf.float64)
    HSOBEL_WEIGHTS = tf.divide(HSOBEL_WEIGHTS, 4)
    HSOBEL_WEIGHTS = tf.expand_dims(HSOBEL_WEIGHTS, 2)
    HSOBEL_WEIGHTS = tf.expand_dims(HSOBEL_WEIGHTS, 3)
    # apply the convolution
    result = tf.nn.conv2d(image, HSOBEL_WEIGHTS, 1, "SAME")
    return result[0, :, :, 0]



