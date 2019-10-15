import warnings
import functools
import operator
from typing import Tuple, TypeVar

import tensorflow as tf

__all__ = [
    'histogram'
]

# Type variable
T = TypeVar('T')

ALLOWABLE_SOURCE_RANGES = ['image', 'dtype']
DEFAULT_NBINS = 256

def get_limits(image: tf.Tensor, source_range: str) -> Tuple[T, T]:
    if source_range == 'image':
        return (tf.reduce_min(image), tf.reduce_max(image))
    elif source_range == 'dtype':
        return image.dtype.limits
        if image.dtype.is_floating:
            return (-1, 1)
        elif image.dtype.is_integer:
            return image.dtype.limits
        else:
            raise TypeError("image must either be an integer of a floating point dtype")
    else:
        raise ValueError("source_range must be in: {}".format(', '.join(ALLOWABLE_SOURCE_RANGES)))

def get_float_centers(limits: Tuple[T, T], nbins: int) -> tf.Tensor:
    """
    Given limits and a number of bins, return the histogram centers.
    
     |  i1  |  i2  |  i3  |
    min                  max
    
    Returns [i1, i2, i3]
    """
    lower, upper = limits
    range_ = upper - lower
    
    # Amount to increase each value by
    increment = range_ / nbins
    # We need to offset the value to get the cneters
    offset_for_center = increment / 2
    return tf.range(start=lower + offset_for_center, limit=upper, delta=increment)

def get_int_centers(limits: Tuple[T, T]) -> tf.Tensor:
    """
    Given limits, return the histogram centers.
    
     i1   i2     i3   ...   iN
     |     |     |     |     |
    min  min+1 min+2  ...   max
    
    Returns [i1, i2, i3, ..., iN]
    """
    lower, upper = limits
    return tf.range(start=lower, limit=upper, delta=tf.constant(1))

def normalize_tensor(tensor: tf.Tensor) -> tf.Tensor:
    return tensor / tf.math.reduce_sum(tensor)

def histogram(image: tf.Tensor, nbins: int=DEFAULT_NBINS, source_range: str='image', normalize: bool=False):
    # TODO - normalize, normalizes the value
    # TODO - cast to int32 and reshape
    
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
    
    image_shape = image.shape
    image = tf.reshape(image, (functools.reduce(operator.mul, image_shape),))
    limits = get_limits(image, source_range)
    if image.dtype.is_integer:
        centers = get_int_centers(limits)
        values = tf.histogram_fixed_width(image, limits, limits[1] - limits[0], dtype=image.dtype)
    else:
        centers = get_float_centers(limits, nbins)
        values = tf.histogram_fixed_width(image, limits, nbins, dtype=image.dtype)
    return values, centers