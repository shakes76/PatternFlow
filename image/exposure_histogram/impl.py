import warnings

import tensorflow as tf

__all__ = [
    'histogram'
]


def get_float_centers(image: tf.Tensor, nbins: int) -> tf.Tensor:
    """
    Given an image and a number of bins, return the histogram centers.
    
     |  i1  |  i2  |  i3  |
    min                  max
    
    Returns [i1, i2, i3]
    """
    lower = image.min()
    upper = image.max()
    range_ = upper - lower
    
    # Amount to increase each value by
    increment = range_ / nbins
    # We need to offset the value to get the cneters
    offset_for_center = increment / 2
    return tf.range(start=lower + offset_for_center, limit=upper, delta=increment)

def get_int_centers(image: tf.Tensor) -> tf.Tensor:
    """
    Given an integer image, return the histogram centers.
    
     i1   i2     i3   ...   iN
     |     |     |     |     |
    min  min+1 min+2  ...   max
    
    Returns [i1, i2, i3, ..., iN]
    """
    lower = image.min()
    upper = image.max()
    return tf.range(start=lower, limit=upper, delta=1)


ALLOWABLE_SOURCE_RANGES = ['image', 'dtype']
DEFAULT_NBINS = 256

def histogram(image: tf.Tensor, nbins: int=DEFAULT_NBINS, source_range: str='image', normalize: bool=False):
    # TODO - source_range takes dtype aswell
    # TODO - normalize
    # h = tf.histogram_fixed_width(uimage, [0, 256], nbins=nbins, dtype=tf.dtypes.int64)
    
    # Check image
    if not isinstance(image, tf.Tensor):
        raise TypeError("image must be of type tf.Tensor")
    if not image.dtype.is_integer and not image.dtype.is_floating:
        raise TypeError("image must either be an integer of a floating point dtype")
    
    # Check nbins
    if not isinstance(nbins, int):
        raise TypeError("nbins must be of type int")
    if nbins <= 0:
        raise ValueError("nbins must be >= 0")
    if nbins != DEFAULT_NBINS and image.dtype.is_integer:
        # Helpful warning if they change nbins and they pass in an integer tensor
        warnings.warn("You have overridden nbins but provided an integer tensor, nbins will be ignored")
    
    # Check source_range
    if source_range not in ALLOWABLE_SOURCE_RANGES:
        raise ValueError("source_range must be in: {}".format(', '.join(ALLOWABLE_SOURCE_RANGES)))
    
    # Don't need to check normalize, as a user could want to pass in a boolean-like
    
    raise NotImplementedError()