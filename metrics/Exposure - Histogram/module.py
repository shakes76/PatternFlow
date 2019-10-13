import tensorflow as tf

tf_int_dtypes = [tf.uint8, tf.uint16, tf.uint32, tf.uint64, tf.int8, tf.int16, tf.int32, tf.int64]

def _tf_is_int_subtype(dtype):
    if dtype in tf_int_dtypes:
        return True
    else:
        return False

def _tf_offset_array(array, low_boundary, high_boundary, sess=None, as_tensor=False):
    """
    offset the array to get the lowest value at 0 if negative
    """
    my_sess = False
    if sess == None:
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
    dtype = image.dtype
    image_max, image_min = dtype.min, dtype.max
    if clip_negative:
        image_min = 0
    return image_max, image_min

def _tf_bincount_histogram(image, source_range, sess=None, as_tensor=False):
    my_sess = False
    if sess == None:
        sess = tf.InteractiveSession()
        my_sess = True
    tf_image = image
    if source_range not in ['image', 'dtype']:
        raise ValueError('Incorrect value for `source_range` argument: {}'.format(source_range))
    if source_range == 'image':
        image_min = tf.cast(tf.math.reduce_min(tf_image), tf.int64).eval()
        image_max = tf.cast(tf.math.reduce_max(tf_image), tf.int64).eval()
    elif source_range == 'dtype':
        image_min, image_max = tf_dtype_limits(tf_image, clip_negative=False)
        
    image, offset = _tf_offset_array(array=image, low_boundary=image_min, high_boundary=image_max, sess=sess, as_tensor=True)
    tf_image = tf.cast(tf.reshape(tensor=tf_image, shape=[-1]), dtype=tf.int32)
    hist = tf.bincount(arr=tf_image, minlength=image_max - image_min + 1, dtype=tf.int32)
    bin_centers = tf.range(start=image_min, limit=image_max + 1)
    if source_range == 'image':
        idx = tf.maximum(image_min, 0)
        hist = hist[idx:]
    
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

def tf_histogram(image, nbins=256, source_range='image', normalize=False, sess=None, as_tensor = False):
    my_sess = False
    if sess == None:
        sess = tf.InteractiveSession()
        my_sess = True
    tf_image = tf.constant(image)
    shape = tf_image.shape
    if len(shape) == 3 and shape[-1] < 4:
        print("This might be a color image. The histogram will be "
             "computed on the flattened image. You can instead "
             "apply this function to each color channel.")
    flatten_image = tf.cast(tf.reshape(tensor=tf_image, shape=[-1]), dtype=tf.int32)
    if _tf_is_int_subtype(flatten_image.dtype):
        hist, bin_centers = _tf_bincount_histogram(image=tf_image, source_range=source_range, sess=sess, as_tensor=True)
    else:
        if source_range == 'image':
            hist_range = None
        elif source_range == 'dtype':
            hist_range = tf_dtype_limits(tf_image, clip_negative=False)
        else:
            ValueError('Wrong value for the `source_range` argument')
        low, high = tf_dtype_limits(tf_image)
        hist = tf.histogram_fixed_width(values=flatten_image, nbins=nbins, value_range=[low, high])
        low, high = float(low), float(high)
        bin_edges = tf.linspace(start=low, stop=high, num=nbins+1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
        
    if normalize:
        hist = hist / tf.math.reduce_sum(hist)

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
