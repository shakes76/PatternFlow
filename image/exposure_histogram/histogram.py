#import numpy as np
from warnings import warn
import tensorflow as tf

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
    See Also
    --------
    cumulative_distribution
    Examples
    --------
    >>> from skimage import data, exposure, img_as_float
    >>> image = img_as_float(data.camera())
    >>> np.histogram(image, bins=2)
    (array([107432, 154712]), array([ 0. ,  0.5,  1. ]))
    >>> exposure.histogram(image, nbins=2)
    (array([107432, 154712]), array([ 0.25,  0.75]))
    """

    
    sess = tf.compat.v1.InteractiveSession()

    image = tf.convert_to_tensor(image)
    print(image.shape)
    

    sh = image.shape
    if len(sh) == 3 and sh[-1] < 4:
        warn("This might be a color image. The histogram will be "
             "computed on the flattened image. You can instead "
             "apply this function to each color channel.")

    flat_image = tf.reshape(image,[-1])
    min = tf.math.reduce_min(flat_image)
    max = tf.math.reduce_max(flat_image)
    # For integer types, histogramming with bincount is more efficient.
    if flat_image.dtype.is_integer:
        hist, bin_centers = _bincount_histogram(flat_image, source_range)
    else:
        if source_range == 'image':
            hist_range = [min, max]
        elif source_range == 'dtype':
            hist_range = dtype_limits(flat_image, clip_negative=False)
        else:
            ValueError('Wrong value for the `source_range` argument')
        #hist, bin_edges = np.histogram(flat_image, bins=nbins, range=hist_range)
        hist = tf.histogram_fixed_width(flat_image, hist_range, nbins=nbins)
        bin_edges = tf.linspace(min,max,nbins+1)

        #https://www.tensorflow.org/api_docs/python/tf/histogram_fixed_width
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        
        tf.global_variables_initializer().run()

    if normalize:
        hist = hist / tf.math.reduce_sum(hist)
    return sess.run(hist), sess.run(bin_centers)




def dtype_limits(image, clip_negative=False):
    """Return intensity limits, i.e. (min, max) tuple, of the image's dtype.
    Parameters
    ----------
    image : ndarray
        Input image.
    clip_negative : bool, optional
        If True, clip the negative range (i.e. return 0 for min intensity)
        even if the image dtype allows negative values.
    Returns
    -------
    imin, imax : tuple
        Lower and upper intensity limits.
    """

#     tf.int8: 8-bit signed integer.
# tf.uint8: 8-bit unsigned integer.
# tf.uint16: 16-bit unsigned integer.
# tf.uint32: 32-bit unsigned integer.
# tf.uint64: 64-bit unsigned integer.
# tf.int16: 16-bit signed integer.
# tf.int32: 32-bit signed integer.
# tf.int64: 64-bit signed integer.
    _integer_types = (np.byte, np.ubyte,          # 8 bits
                  np.short, np.ushort,        # 16 bits
                  np.intc, np.uintc,          # 16 or 32 or 64 bits
                  np.int_, np.uint,           # 32 or 64 bits
                  np.longlong, np.ulonglong)  # 64 bits
    _integer_ranges = {t: (np.iinfo(t).min, np.iinfo(t).max)
                   for t in _integer_types}
    dtype_range = {np.bool_: (False, True),
               np.bool8: (False, True),
               np.float16: (-1, 1),
               np.float32: (-1, 1),
               np.float64: (-1, 1)}
    dtype_range.update(_integer_ranges)

    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax

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
        image_min = np.min(image).astype(np.int64)
        image_max = np.max(image).astype(np.int64)
    elif source_range == 'dtype':
        image_min, image_max = dtype_limits(image, clip_negative=False)
    image, offset = _offset_array(image, image_min, image_max)
    hist = np.bincount(image.ravel(), minlength=image_max - image_min + 1)
    #https://www.tensorflow.org/api_docs/python/tf/math/bincount
    bin_centers = np.arange(image_min, image_max + 1)
    if source_range == 'image':
        idx = max(image_min, 0)
        hist = hist[idx:]
    return hist, bin_centers

def _offset_array(arr, low_boundary, high_boundary):

    """Offset the array to get the lowest value at 0 if negative.
    """
    if low_boundary < 0:
        offset = low_boundary
        dyn_range = high_boundary - low_boundary
        # get smallest dtype that can hold both minimum and offset maximum
        offset_dtype = np.promote_types(np.min_scalar_type(dyn_range),
                                        np.min_scalar_type(low_boundary))
        if arr.dtype != offset_dtype:
            # prevent overflow errors when offsetting
            arr = arr.astype(offset_dtype)
        arr = arr - offset
    else:
        offset = 0
    return arr, offset