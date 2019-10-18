import tensorflow as tf
import warnings

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
    image = image.flatten()
    
    # For integer types, histogramming with bincount is more efficient.
    if image.dtype == 'int':
        hist, bin_centers = _bincount_histogram(image, source_range)
        
    else:
        if source_range == 'image':
            hist_range = None
        elif source_range == 'dtype':
            hist_range = dtype_limits(image, clip_negative=False)
        else:
            ValueError('Wrong value for the `source_range` argument')
        hist, bin_edges = np.histogram(image, bins=nbins, range=hist_range)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.

    if normalize:
        hist = hist / np.sum(hist)
    return hist, bin_centers