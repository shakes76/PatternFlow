"""
        COMP3710 Open source project
        Name: Puyang Deng
        Student Number: s44430487
        
        Tensorflow port of skimage.exposure.histogram
"""
import tensorflow as tf
import warnings

dtype_range = {tf.bool: (False, True),
               tf.float16: (-1, 1),
               tf.float32: (-1, 1),
               tf.float64: (-1, 1)}

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
    
    imin, imax = dtype_range[image.dtype.type]
    if clip_negative:
        imin = 0
    return imin, imax

def _offset_array(arr, low_boundary, high_boundary):
    
    """Offset the array to get the lowest value at 0 if negative.
    Parameters
    ----------
    arr : ndarray
        Input image or array.
    low_boundary : integer
        The lower boundary intensity limit of the input image.
    high_boundary : integer
        The lower boundary intensity limit of the input image.
    Returns
    -------
    arr, offset : tuple
        Original input array and offset array to origin.
    """
    
    if low_boundary < 0:
        offset = low_boundary
        arr = arr - offset
    else:
        offset = 0
    return arr, offset

def _bincount_histogram(image, source_range):
    
    """Efficient histogram calculation for an image of integers.
    This function is significantly more efficient than self built histogram 
    but works only on images of integers. It is based on self built bincount.
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
    
    if source_range not in ['image']:
        raise ValueError('Incorrect value for `source_range` argument: {}'.format(source_range))
    
    if source_range == 'image':
        image_min = int(image.min().astype(tf.int64))
        image_max = int(image.max().astype(tf.int64))
        
    elif source_range == 'dtype':
        image_min, image_max = dtype_limits(image, clip_negative=False)#define the data max and min range
        
    image, offset = _offset_array(image, image_min, image_max)
    minlength=image_max - image_min + 1
    
    hist = tf.math.bincount(image.ravel(), minlength)
    hist_centers = tf.range(image_min,minlength)
    
    if source_range == 'image':
        idx = max(image_min, 0)
        hist = hist[idx:]
        
    return hist, hist_centers

def histogram(image, nbins=256, source_range='image', normalize=False):
    
    """Return histogram of image.
    This function returns the centers of bins and does not rebin integer 
    arrays. For integer arrays, each integer value has
    its own bin, which improves speed and intensity-resolution with self built
    bincount_histogram function.
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
    hist_centers : array
        The values at the center of the bins.
    """
   
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
            hist_range = [0.0, 256.0]#modify the historgram range as image RGB 256   
        elif source_range == 'dtype':
            hist_range = dtype_limits(image, clip_negative=False)#modify the histogram range as datype case
        else:
            ValueError('Wrong value for the `source_range` argument')
            
        hist_centers = [i for i in range(int(hist_range[1]))]
        
        tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        hist = tf.histogram_fixed_width(tensor, hist_range, nbins)
            
    if normalize:
        hist = hist / tf.reduce_sum(hist)
        
    return hist, hist_centers
