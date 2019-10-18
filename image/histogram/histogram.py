import tensorflow as tf
import warnings

dtype_range = {tf.bool: (False, True),
               tf.float16: (-1, 1),
               tf.float32: (-1, 1),
               tf.float64: (-1, 1)}

def dtype_limits(image, clip_negative=False):
    
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax

def _offset_array(arr, low_boundary, high_boundary):
    """Offset the array to get the lowest value at 0 if negative."""
    if low_boundary < 0:
        offset = low_boundary
        arr = arr - offset
    else:
        offset = 0
    return arr, offset

def _bincount_histogram(image, source_range):
    
    if source_range not in ['image']:
        raise ValueError('Incorrect value for `source_range` argument: {}'.format(source_range))
    
    if source_range == 'image':
        image_min = int(image.min().astype(tf.int64))
        image_max = int(image.max().astype(tf.int64))
        
    elif source_range == 'dtype':
        image_min, image_max = dtype_limits(image, clip_negative=False)
        
    image, offset = _offset_array(image, image_min, image_max)
    minlength=image_max - image_min + 1
    
    hist = tf.math.bincount(image.ravel(), minlength)
    bin_centers = tf.range(image_min,minlength)
    
    if source_range == 'image':
        idx = max(image_min, 0)
        hist = hist[idx:]
        
    return hist, bin_centers

def histogram(image, nbins=256, source_range='image', normalize=False):
   
    sh = tf.shape(image)
    
    if len(sh) == 3 and sh[-1] < 4:
        warnings.warn("This might be a color image. The histogram will be "
             "computed on the flattened image. You can instead "
             "apply this function to each color channel.")
        
    image = image.flatten()#flattern the image into 1 dimentional vector
    
    # For integer types, histogramming with bincount is more efficient.
    if image.dtype == 'int':
        hist, bin_centers = _bincount_histogram(image, source_range)
        
    else:
        
        if source_range == 'image':
            hist_range = [0.0, 256.0]
        elif source_range == 'dtype':
            hist_range = dtype_limits(image, clip_negative=False)
        else:
            ValueError('Wrong value for the `source_range` argument')
            
        hist_centers = [i for i in range(256)]
        
        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        hist = tf.histogram_fixed_width(tensor, hist_range, nbins)
            
    if normalize:
        hist = hist / np.sum(hist)
        
    return hist, bin_centers