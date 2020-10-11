import tensorflow as tf
import tensorflow_probability as tfp

def equalize_hist(image, nbins=256, mask=None):
    """Returns an image after histogram equalisation

    Parameters
    -----------
    image : array
        Image to be equalised
    nbins : int optional
        Number of bins for the histogram
    mask: array optional
        Array of bools (as 1s & 0s) which restricts the areas used
        to calculate the histogram

    Returns:
    -----------
    output : array
        Float32 array representing the equalised image

    [References]
    https://github.com/scikit-image/scikit-image/blob/master/skimage/exposure/exposure.py#L187
    http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    """

    dims = image.shape

    with tf.Session() as sess:

        # Create and initialise variables
        image_tensor = tf.Variable(image, dtype=tf.float32)
        x_min = tf.constant(0, dtype=tf.float32)
        x_max = tf.constant(255, dtype=tf.float32)
        values_range = tf.constant([0., 255.], dtype=tf.float32)

        if mask is not None:
            mask_tensor = tf.Variable(mask)

        sess.run(tf.global_variables_initializer())

        # Flatten image
        image_tensor = tf.reshape(image_tensor, [-1])

        # Calculate histogram and cumulative distribution
        if mask is not None:
            mask_tensor = tf.reshape(mask_tensor, [-1])
            histogram = tf.histogram_fixed_width(tf.boolean_mask(image_tensor, mask_tensor), values_range, nbins)
        else:
            histogram = tf.histogram_fixed_width(image_tensor, values_range, nbins)

        cdf = tf.cumsum(histogram)

        # Normalise cumulative distribution
        cdf = tf.divide(cdf, tf.gather(cdf, nbins-1))

        # Calculate equalised image
        image_eq = tfp.math.interp_regular_1d_grid(image_tensor, x_min, x_max, tf.cast(cdf, tf.float32))
        image_eq = tf.reshape(image_eq, dims)
        image_eq = sess.run(image_eq)

        sess.close()

        return image_eq