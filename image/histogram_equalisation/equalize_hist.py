import tensorflow as tf
import tensorflow_probability as tfp

def equalize_hist(image, nbins=256, mask=None):

    # Scikit Implementation
    # https://github.com/scikit-image/scikit-image/blob/master/skimage/exposure/exposure.py#L187

    # if mask is not None:
    #     mask = np.array(mask, dtype=bool)
    #     cdf, bin_centers = cumulative_distribution(image[mask], nbins)
    # else:
    #     cdf, bin_centers = cumulative_distribution(image, nbins)
    # out = np.interp(image.flat, bin_centers, cdf)
    # return out.reshape(image.shape)

    # Jan Erik Solem's Implementation
    # http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # imhist, bins = histogram(im.flatten(), nbr_bins, normed=True)
    # cdf = imhist.cumsum()  # cumulative distribution function
    # cdf = 255 * cdf / cdf[-1]  # normalize
    #
    # # use linear interpolation of cdf to find new pixel values
    # im2 = interp(im.flatten(), bins[:-1], cdf)
    #
    # return im2.reshape(im.shape), cdf

    dims = image.shape

    with tf.Session() as sess:
        # Initialise variables
        image_tf = tf.Variable(image, dtype=tf.float32)
        values_range = tf.constant([0., 255.], dtype=tf.float32)
        x_min = tf.constant(0, dtype=tf.float32)
        x_max = tf.constant(255, dtype=tf.float32)
        sess.run(tf.global_variables_initializer())

        # Flatten image
        image_tf = tf.reshape(image_tf, [-1])

        # Calculate cumulative distribution
        histogram = tf.histogram_fixed_width(image_tf, values_range, nbins)
        cdf = tf.cumsum(histogram)


        # Calculate equalised image
        image_eq = tfp.math.interp_regular_1d_grid(tf.cast(image_tf, tf.float32), x_min, x_max, tf.cast(cdf, tf.float32))

        # if dims[2]:
        #     image_eq = tf.reshape(image_eq, [dims[0], dims[1], dims[2]])
        # else:
        #     image_eq = tf.reshape(image_eq, [dims[0], dims[1]])

        image_eq = tf.reshape(image_eq, dims)
        image_eq = sess.run(image_eq)

        sess.close()

        return image_eq