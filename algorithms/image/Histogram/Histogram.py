import tensorflow as tf

"""
        COMP3710 Open source project
        Name: Chrisbin James
        Student Number: 45431780
        
        Tensorflow port of skimage.exposure.histogram
"""

# List of integer tensorflow datatypes
tf_int_dtypes = [tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int8, tf.int16, tf.int32, tf.int64]

def _tf_is_int_subtype(dtype):
    """
    Checks whether a given datatype is a subset of integer.
    
    Args
    ---------------
    dtype: A tensorflow dtype
    
    Returns
    ---------------
    Boolean: True if the data type is a subset of integer, False other wise
    """
    # check if given dtype in list of integer dtypes
    if dtype in tf_int_dtypes:
        return True
    else:
        return False

def _tf_offset_array(array, low_boundary, high_boundary, sess=None, as_tensor=False):
    """
    Offsets a given array to get the lowest value to 0 if the lowest value is negative.
    
    Args
    ---------------
    array: An array (non - tensor)
    low_boundary: The lowest boundary value of the array.
    high_boundary: The highest boundary value of the array.
    sess: bool, optional A tensorflow session.
    as_tensor: bool, optional returns the result as a tensor if true otherwise returns the result as evaluated values.
               default value is false
    
    Returns
    ---------------
    array, offset: tensor/array, tensor/int/float The modified array with lowest value shifted to 0
    and the value by which the entire array is shifted
    """
    # check if a running tensorflow session is provided
    my_sess = False
    if sess == None:
        # if not initialize a tensorflow session
        sess = tf.InteractiveSession()
        my_sess = True
        
    # convert given variable into a tensor variable
    tf_array = tf.Variable(array)
    tf.global_variables_initializer().run()
    # check if low boundary is lower than 0
    if low_boundary < 0:
        # get the smallest data type that can hold minimum and offset maximum and
        # cast the given array into the aforementioned data type
        offset = tf.constant(low_boundary)
        dyn_range = high_boundary - offset
        if tf_array.dtype != dyn_range.dtype:
            tf_array = tf.cast(tf_array, dyn_range.dtype)
        # offset the array
        tf_array = tf_array - offset
    else:
        offset = tf.constant(0)
    
    # check if results  need to be returned as tensors and close the tensorflow session if it was initialized by this function
    if as_tensor:
        if my_sess:
            sess.close()
            return tf_array, offset
        else:
            return tf_array, offset
    else:
        if my_sess:
            array, offset = tf_array.eval(), offset.eval()
            sess.close()
            return array, offset
        else:
            array, offset = tf_array.eval(), offset.eval()
            return array, offset

def tf_dtype_limits(image, clip_negative = False):
    """
    For a given image tensor returns the intensity limits, i.e. (min, max) tuple, of the image's dtype.
    
    Args
    ---------------
    image: tensor, A tensor of a image
    clip_negative: bool, optional If True, clip the negative range (i.e. return 0 for min intensity)
                   even if the image dtype allows negative values. Default value is False
    
    Returns
    ---------------
    image_min, image_max : tuple Lower and upper intensity limits.
    """
    # get image datatype
    dtype = image.dtype
    # get max and min values
    image_max, image_min = dtype.min, dtype.max
    # clip negative range if required
    if clip_negative:
        image_min = 0
    # return values
    return image_max, image_min

def _tf_bincount_histogram(image, source_range, sess=None, as_tensor=False):
    """
    Efficient histogram calculation for an image of integers.
    This function is significantly more efficient than tf.histogram_fixed_width but works only on images of integers.
    It is based on tf.bincount.
    
    Args
    ---------------
    image: image: tensor, A tensor of a image
    source_range : string
        'image' determines the range from the input image.
        'dtype' determines the range from the expected range of the images of that data type.
    sess: bool, optional A tensorflow session.
    as_tensor: bool, optional returns the result as a tensor if true otherwise returns the result as evaluated values.
               default value is false
    
    Returns
    ---------------
    hist : array/tensor The values of the histogram.
    bin_centers : array/tensor The values at the center of the bins. 
    """
    # check if a tensorflow session is provided
    my_sess = False
    if sess == None:
        # if not initialize a tensorflow session
        sess = tf.InteractiveSession()
        my_sess = True
    tf_image = image
    # Determine how to calculate value range for the histogram
    if source_range not in ['image', 'dtype']:
        raise ValueError('Incorrect value for `source_range` argument: {}'.format(source_range))
    if source_range == 'image':
        # get value range from image
        image_min = tf.cast(tf.math.reduce_min(tf_image), tf.int64).eval()
        image_max = tf.cast(tf.math.reduce_max(tf_image), tf.int64).eval()
    elif source_range == 'dtype':
        # get value range from image datatype
        image_min, image_max = tf_dtype_limits(tf_image, clip_negative=False)
    # offset the image array to get low value bolundary to zero    
    image, offset = _tf_offset_array(array=image, low_boundary=image_min, high_boundary=image_max, sess=sess, as_tensor=True)
    # flatten the image
    tf_image = tf.cast(tf.reshape(tensor=tf_image, shape=[-1]), dtype=tf.int32)
    # get the bincount value
    hist = tf.bincount(arr=tf_image, minlength=image_max - image_min + 1, dtype=tf.int32)
    # get bin centers
    bin_centers = tf.range(start=image_min, limit=image_max + 1)
    # if value range is calculated via image
    if source_range == 'image':
        # get the min value in image
        idx = tf.maximum(image_min, 0)
        hist = hist[idx:]
    # check if results  need to be returned as tensors and close the tensorflow session if it was initialized by this function
    if as_tensor:
        if my_sess:
            sess.close()
            return hist, bin_centers
        else:
            return hist, bin_centers
    else:
        if my_sess:
            hist, bin_centers = hist.eval(), bin_centers.eval()
            sess.close()
            return hist, bin_centers
        else:
            hist, bin_centers = hist.eval(), bin_centers.eval()
            return hist, bin_centers

def histogram(image, nbins=256, source_range='image', normalize=False, sess=None, as_tensor = False):
    """
    Returns the histogram of a an image. This function returns the centers of bins and
    does not rebin integer arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution.

    The histogram is computed on the flattened image: for color images, the
    function should be used separately on each channel to obtain a histogram
    for each color channel.
    
    Args
    ---------------
    image : array Input image.
    nbins : int, optional Number of bins used to calculate histogram. This value is ignored for integer arrays.
    source_range : string, optional
        'image' (default) determines the range from the input image.
        'dtype' determines the range from the expected range of the images of that data type.
    normalize : bool, optional If True, normalize the histogram by the sum of its values.
    sess: bool, optional A tensorflow session.
    as_tensor: bool, optional returns the result as a tensor if true otherwise returns the result as evaluated values.
               default value is false
    
    Returns
    ---------------
    hist : array/tensor The values of the histogram.
    bin_centers : array/tensor The values at the center of the bins.
    """
    # check if a tensorflow session is provided
    my_sess = False
    if sess == None:
        # if not initialize a tensorflow session
        sess = tf.InteractiveSession()
        my_sess = True
    # convert image to tensor
    tf_image = tf.constant(image)
    # check if a color image/ multiple channels for the image is provided
    shape = tf_image.shape
    if len(shape) == 3 and shape[-1] < 4:
        print("This might be a color image. The histogram will be "
             "computed on the flattened image. You can instead "
             "apply this function to each color channel.")
    # check if image is an interger array
    if _tf_is_int_subtype(tf_image.dtype):
        # calculate histogram using bincount
        flatten_image = tf.cast(tf.reshape(tensor=tf_image, shape=[-1]), dtype=tf.int32)
        hist, bin_centers = _tf_bincount_histogram(image=tf_image, source_range=source_range, sess=sess, as_tensor=True)
    else:
        # if not calculate histogram normally
        flatten_image = tf.cast(tf.reshape(tensor=tf_image, shape=[-1]), dtype=tf.float32)
        # determine how to calculate histogram value range
        if source_range == 'image':
            # calculate range via image values
            low, high = tf.math.reduce_min(tf_image).eval(), tf.math.reduce_max(tf_image).eval()
            hist_range = [low, high]
        elif source_range == 'dtype':
            # calculate range via image datatype limits
            hist_range = tf_dtype_limits(tf_image, clip_negative=False)
        else:
            ValueError('Wrong value for the `source_range` argument')
        # get histogram            
        hist = tf.histogram_fixed_width(values=flatten_image, nbins=nbins, value_range=hist_range)
        # get histogram bins
        bin_edges = tf.linspace(start=low, stop=high, num=nbins+1)
        # calculate bin centers
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    # check if histogram needs to be normalized    
    if normalize:
        hist = hist / tf.math.reduce_sum(hist)
    # check if results  need to be returned as tensors and close the tensorflow session if it was initialized by this function
    if as_tensor:
        if my_sess:
            sess.close()
            return hist, bin_centers
        else:
            return hist, bin_centers
    else:
        if my_sess:
            hist, bin_centers = hist.eval(), bin_centers.eval()
            sess.close()
            return hist, bin_centers
        else:
            hist, bin_centers = hist.eval(), bin_centers.eval()
            return hist, bin_centers
