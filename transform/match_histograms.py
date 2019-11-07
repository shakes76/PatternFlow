#!/usr/bin/env python
# -*- coding: utf-8 -*-
# last updated:2019-11-07
"""
Algorithm from skimage.exposure model in Tensorflow.
"""
__author__ = "Yao Chen"

import tensorflow as tf 
import cProfile

def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    parameters:
    ----------
    source: tensor
    template: tensor

    """
    #reshape the data 
    source_flatten = tf.reshape(source,[-1])
    template_flatten = tf.reshape(template,[-1])
    #convert type to int 64
    source_flatten1 = tf.cast(source_flatten ,dtype =tf.int64)
    source_flatten1 = tf.sort(source_flatten1)
    template_flatten = tf.cast(template_flatten ,dtype =tf.int64)
    template_flatten = tf.sort(template_flatten)

    src_values, src_unique_indices, src_counts = tf.unique_with_counts(source_flatten1)
    src_indice = unique(source_flatten)
    tmpl_values, tmpl_counts = tf.unique_with_counts(template_flatten)

    # change to tensor 
    source_size = tf.size(source_flatten)
    template_size = tf.size(template_flatten)
    # calculate normalized quantiles for each array
    src_quantiles = tf.cumsum(src_counts) / source_size
    tmpl_quantiles = tf.cumsum(tmpl_counts) / template_size

    interp_a_values = interpolate(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)

def  interpolate( dx, dy, x, name='interpolate' ):
    
    # get the tensor varaiable
    with tf.variable_scope(name):
        with tf.variable_scope('neighbors'):
            delVals = dx - x
            ind_1   = tf.argmax(tf.sign( delVals ))
            ind_0   = ind_1 - 1

        with tf.variable_scope('calculation'):
            value   = tf.cond( x[0] <= dx[0], lambda : dy[:1], lambda : tf.cond( x[0] >= dx[-1], 
                                     lambda : dy[-1:], lambda : (dy[ind_0] + 
                                     (dy[ind_1] - dy[ind_0]) *(x-dx[ind_0])/ (dx[ind_1]-dx[ind_0]))
                             ))
        #interpolate result                  
        result = tf.multiply(value[0], 1, name='y')

    return result

def unique(x):
    """ 
    A funtion to replace np.unique
    function can return to the index of list with unique value from huge to small.
    """
    x = tf.cast(x,dtype = tf.int64)
    # sort the tensor
    y = tf.argsort(x,stable = None)
    #create array
    array = x[y]
    tensor = tf.zeros(array.shape, dtype=tf.bool)
    # change it to tensor
    k = tf.cast(tensor, dtype = tf.int64)
    # sum
    index = tf.zeros(tensor.shape, dtype=tf.int64)
    index[y] = tf.cumsum(k) - 1
    # convert to tensor
    index = tf.convert_to_tensor(index)
    return index

def match_histograms(image, reference, *, multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    Parameters
    ----------
    image : tensor
        Input image. Can be gray-scale or in color.
    reference : tensor
        Image to match histogram of. Must have the same number of channels as
        image.
    multichannel : bool, optional
        Apply the matching separately for each channel.
    Returns
    -------
    matched : tensor
        Transformed input image.
    Raises
    ------
    ValueError
        Thrown when the number of channels in the input image and the reference
        differ.
    References
    ----------
    .. [1] http://paulbourke.net/miscellaneous/equalisation/
    """
    # make sure two image has same channels.
    if tf.rank(image) != tf.rank(reference):
        raise ValueError('Image and reference must have the same number '
                         'of channels.')
    # when channel multiply
    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and '
                             'reference image must match!')

        matched = tf.zeros(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel],
                                                    reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        # redraw the image
        matched = _match_cumulative_cdf(image, reference)

    return matched