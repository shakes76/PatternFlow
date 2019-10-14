import tensorflow as tf


def adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False):
    """Performs Sigmoid Correction on the input image (tensorflow version).
    Also known as Contrast Adjustment.
    This function transforms the input image pixelwise according to the
    equation ``O = 1/(1 + exp*(gain*(cutoff - I)))`` after scaling each pixel
    to the range 0 to 1.
    Parameters
    ----------
    image : ndarray
        Input image.
    cutoff : float, optional
        Cutoff of the sigmoid function that shifts the characteristic curve
        in horizontal direction. Default value is 0.5.
    gain : float, optional
        The constant multiplier in exponential's power of sigmoid function.
        Default value is 10.
    inv : bool, optional
        If True, returns the negative sigmoid correction. Defaults to False.
    Returns
    -------
    out : tensor
        Sigmoid corrected output image.
    """
    # Transform the ndarray to a tensor
    image_tensor = tf.constant(image)

    # raising InvalidArgumentError unless image_tensor is all non-negative
    with tf.control_dependencies([tf.assert_non_negative(image_tensor)]):
        dtype = image_tensor.dtype

        # scale pixel values to [0, 1], which support a variety of data types.
        scale = float(dtype.limits[1] - dtype.limits[0])
        image_tensor = tf.cast(image_tensor, tf.float32)

        # equation: 1/(1 + exp*(gain*(cutoff - I)))
        if inv:
            out = (1 - 1 / (1 + tf.math.exp(
                gain * (cutoff - tf.div(image_tensor, scale))))) * scale
        else:
            out = (1 / (1 + tf.math.exp(
                gain * (cutoff - tf.div(image_tensor, scale))))) * scale

        #  convert to original type
        out = tf.cast(out, dtype)

        with tf.Session() as sess:
            return sess.run(out)




