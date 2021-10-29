
from warnings import warn
import tensorflow as tf


"""
    COMP3710 Report - Algorithm Implementation

    Student: Umberto Pietroni 45981427

    Porting of skimage.exposure.histogram algorithm to Tensorflow
    https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/exposure/exposure.py#L77

"""

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
    
    sh = image.shape

    if len(sh) == 3 and sh[-1] < 4:
        warn("This might be a color image. The histogram will be "
             "computed on the flattened image. You can instead "
             "apply this function to each color channel.")
    #flat the image
    flat_image = tf.reshape(image,[-1])     

    # For integer types, histogramming with bincount is more efficient.
    if flat_image.dtype.is_integer:
        hist, bin_centers = _bincount_histogram(flat_image, source_range)
    else:
        if source_range == 'image':
            min = tf.math.reduce_min(flat_image)
            max = tf.math.reduce_max(flat_image)
            hist_range = [min, max]
        elif source_range == 'dtype':
            hist_range = dtype_limits(flat_image, clip_negative=False)
        else:
            ValueError('Wrong value for the `source_range` argument')
       
        #https://www.tensorflow.org/api_docs/python/tf/histogram_fixed_width
        hist = tf.histogram_fixed_width(flat_image, hist_range, nbins=nbins)
        min,max = hist_range
        #bins of tf.histogram_fixed_width are equal width and determined by the arguments hist_range and nbins
        bin_edges = tf.linspace(min,max,nbins+1)

        #compute the centers of bin
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        
        tf.compat.v1.global_variables_initializer()

    if normalize:
        hist = hist / tf.math.reduce_sum(hist)
    
    ret_hist =  sess.run(hist)
    ret_bin_centers = sess.run(bin_centers)
    sess.close()
    return ret_hist, ret_bin_centers




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

    https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/util/dtype.py#L35
    """

    _integer_types = (tf.int8, tf.uint8,          # 8 bits
                  tf.int16, tf.uint16,        # 16 bits
                  tf.int32, tf.uint32,          #32 bits
                  tf.int64, tf.uint64)  # 64 bits

    _integer_ranges = {t: (t.min, t.max)
                   for t in _integer_types}
    dtype_range = {tf.bool: (False, True),
               tf.float16: (-1.0, 1.0),
               tf.float32: (-1.0, 1.0),
               tf.float64: (-1.0, 1.0)}
    dtype_range.update(_integer_ranges)

    imin, imax = dtype_range[image.dtype]
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

    https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/exposure/exposure.py#L38
    """

  
    if source_range not in ['image', 'dtype']:
        raise ValueError('Incorrect value for `source_range` argument: {}'.format(source_range))
    if source_range == 'image':
        image_min = tf.math.reduce_min(image)
        image_max = tf.math.reduce_max(image)

    elif source_range == 'dtype':
        image_min, image_max = dtype_limits(image, clip_negative=False)

    #cast to int32 for tf.math.bincount
    min = tf.dtypes.cast(image_min, tf.int32)
    max = tf.dtypes.cast(image_max, tf.int32)
    image_to_int = tf.dtypes.cast(image, tf.int32)

    image_2 = _offset_array(image_to_int, min, max)
    flat_image = tf.reshape(image_2,[-1])
       
    leng = tf.math.subtract(max,min)
    hist = tf.math.bincount(flat_image, minlength=leng + 1)
    #https://www.tensorflow.org/api_docs/python/tf/math/bincount
    
    
    bin_centers = tf.range(min, max + 1)
    tf.compat.v1.global_variables_initializer()

    
    if source_range == 'image':
        idx = tf.math.maximum(min, 0)
        hist = hist[idx:]
    return hist, bin_centers

def _offset_array(arr, low_boundary, high_boundary):

    """Offset the array to get the lowest value at 0 if negative.

    https://github.com/scikit-image/scikit-image/blob/v0.15.0/skimage/exposure/exposure.py#L38
    """
    def true_cond(arr,low_boundary,high_boundary):
        offset = low_boundary
        dyn_range = high_boundary - low_boundary

        # prevent overflow errors when offsetting
        if not arr.dtype.is_compatible_with(dyn_range.dtype):
            tf.dtypes.cast(arr,dyn_range.dtype)
        s_arr = tf.math.subtract(arr,offset)
        return s_arr
    def false_cond(arr):
        offset = tf.constant([0])
        return arr

    res = tf.cond(low_boundary < 0,lambda: true_cond(arr,low_boundary,high_boundary),lambda: false_cond(arr))

    return res