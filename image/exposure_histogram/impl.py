import tensorflow as tf

__all__ = [
    'histogram'
]


def get_centers(image, nbins):
    """
    Given an image and a number of bins, return the histogram centers.
    
     |  i1  |  i2  |  i3  |
    min                  max
    
    Returns [i1, i2, i3, ...]
    """
    lower = image.min()
    upper = image.max()
    range_ = upper - lower
    
    # Amount to increase each value by
    increment = range_ / nbins
    # We need to offset the value to get the cneters
    offset_for_center = increment / 2
    return tf.range(start=lower + offset_for_center, limit=upper, delta=increment)

def histogram(image, nbins=256, source_range='image', normalize=False):
    # h = tf.histogram_fixed_width(uimage, [0, 256], nbins=nbins, dtype=tf.dtypes.int64)
    ...