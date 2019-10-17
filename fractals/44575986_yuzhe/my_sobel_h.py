import tensorflow as tf
import numpy as np


def tf_sobel_h(image, mask=None):

    """Find the vertical edges of an image using the Sobel transform.
    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.
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

    image = tf.expand_dims(image, 0)
    image = tf.expand_dims(image, 3)
    HSOBEL_WEIGHTS = tf.Variable([[1, 2, 1],
                                   [0, 0, 0],
                                    [-1, -2, -1]], dtype=tf.float64)
    HSOBEL_WEIGHTS = tf.divide(HSOBEL_WEIGHTS, 4)
    HSOBEL_WEIGHTS = tf.expand_dims(HSOBEL_WEIGHTS, 2)
    HSOBEL_WEIGHTS = tf.expand_dims(HSOBEL_WEIGHTS, 3)

    result = tf.nn.conv2d(image, HSOBEL_WEIGHTS, 1, "SAME")

    if mask is None:
        result = result[0, :].assign(0)
        result = result[-1, :].assign(0)
        result = result[:, 0].assign(0)
        result = result[:, -1].assign(0)
        return result
