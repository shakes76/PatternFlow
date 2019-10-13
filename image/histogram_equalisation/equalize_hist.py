import tensorflow as tf

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

    with tf.Session as sess:
        sess.close()