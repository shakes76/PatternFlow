import matplotlib.pyplot as plt
import tensorflow as tf
from skimage import img_as_ubyte
from skimage import data

def _offset_array(arr, low_boundary, high_boundary):
    """Offset the array to get the lowest value at 0 if negative."""
    if low_boundary < 0:
        offset = low_boundary
        dyn_range = high_boundary - low_boundary
        arr = arr - offset
    else:
        offset = 0
    return arr, offset


def _bincount_histogram(image, source_range):
    """
    Efficient histogram calculation for an image of integers.
    This function is significantly more efficient than np.histogram but
    works only on images of integers. It is based on np.bincount.
    Parameters
    ----------
    image : array
        Input image.
    source_range : string
        'image' determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    """
    if source_range not in ['image', 'dtype']:
        raise ValueError('Incorrect value for `source_range` argument: {}'.format(source_range))
    if source_range == 'image':
        image_min = tf.reduce_min(image)
        image_max = tf.reduce_max(image)
        with tf.Session() as sess:
            image_min = sess.run(image_min)
            image_max = sess.run(image_max)
#     elif source_range == 'dtype':
#         image_min, image_max = dtype_limits(image, clip_negative=False)
    image, offset = _offset_array(image, image_min, image_max)#image has already been offset at this moment
    values = tf.constant(image.ravel())
    with tf.Session() as sess:
        values = sess.run(values)
    hist = tf.math.bincount(values) 
    with tf.Session() as sess:
        hist = sess.run(hist)
#     hist = np.bincount(image.ravel(), minlength=image_max - image_min + 1)
#     bin_centers = np.arange(image_min, image_max + 1)
    start = int(image_min)
    limit = int(image_max+1)
    delta = 1
    # use tf.range() to represent np.arange()
    bin_centers = tf.range(start, limit, delta) 
    with tf.Session() as sess:
        bin_centers = sess.run(bin_centers)
    if source_range == 'image':
        idx = max(image_min, 0)
        hist = hist[idx:]
    return hist, bin_centers


def histogram(image, nbins=256, source_range='image', normalize=False):
    """Return histogram of image.
    Unlike `numpy.histogram`, this function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.
    The histogram is computed on the flattened image: for color images, the
    function should be used separately on each channel to obtain a histogram
    for each color channel.
    Parameters
    ----------
    image : array
        Input image.
    nbins : int, optional
        Number of bins used to calculate histogram. This value is ignored for
        integer arrays.
    source_range : string, optional
        'image' (default) determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    normalize : bool, optional
        If True, normalize the histogram by the sum of its values.
    Returns
    -------
    hist : array
        The values of the histogram.
    bin_centers : array
        The values at the center of the bins.
    """
    sh = image.shape
    if len(sh) == 3 and sh[-1] < 4:
        # This function doesn't directly return the histograms for colored images
        warn("This might be a color image. The histogram will be "
             "computed on the flattened image. You can instead "
             "apply this function to each color channel.")
    # The image dtype has already been converted as tf.int32 in the main.py file
    image = image.flatten()
    if source_range == 'image':
        hist_range = None
    elif source_range == 'dtype':
        hist_range = dtype_limits(image, clip_negative=False)
    else:
        ValueError('Wrong value for the `source_range` argument')
    hist, bin_centers = _bincount_histogram(image, source_range)

    if normalize:
        # Use tf.reduce_sum to represent numpy.sum()
        hist_sum = tf.reduce_sum(hist)
        with tf.Session() as sess:
            hist = sess.run(hist_sum)
        hist = hist / hist_sum
        
    return hist, bin_centers