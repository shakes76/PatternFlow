import tensorflow as tf

def _tf_offset_array(array, low_boundary, high_boundary, as_tensor=False):
    """
    offset the array to get the lowest value at 0 if negative
    """
    sess = tf.InteractiveSession()
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
        sess.close()
        return tf_array, offset
    else:
        array, offset = tf_array.eval(), offset.eval()
        sess.close()
        return array, offset

def tf_dtype_limits(image, clip_negative = False):
    dtype = image.dtype
    image_max, image_min = dtype.min, dtype.max
    if clip_negative:
        image_min = 0
    return image_max, image_min

def _tf_bincount_histogram(image, source_range, as_tensor=False):
    sess = tf.InteractiveSession()
    tf_image = tf.constant(image)
    if source_range not in ['image', 'dtype']:
        raise ValueError('Incorrect value for `source_range` argument: {}'.format(source_range))
    if source_range == 'image':
        image_min = tf.cast(tf.math.reduce_min(tf_image), tf.int64).eval()
        image_max = tf.cast(tf.math.reduce_max(tf_image), tf.int64).eval()
    elif source_range == 'dtype':
        image_min, image_max = dtype_limits(image, clip_negative=False)
        
    image, offset = _tf_offset_array(array=image, low_boundary=image_min, high_boundary=image_max, as_tensor=True)
    tf_image = tf.cast(tf.reshape(tensor=tf_image, shape=[-1]), dtype=tf.int32)
    hist = tf.bincount(arr=tf_image, minlength=image_max - image_min + 1, dtype=tf.int32)
    bin_centers = tf.range(start=image_min, limit=image_max + 1)
    if source_range == 'image':
        idx = tf.maximum(image_min, 0)
        hist = hist[idx:]
    
    if as_tensor:
        sess.close()
        return hist, bin_centers
    else:
        hist, bin_centers = hist.eval(), bin_centers.eval()
        sess.close()
        return hist, bin_centers
