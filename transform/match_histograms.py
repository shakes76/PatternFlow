import tensorflow as tf 

def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    #reshape the data 
    source_flatten = tf.reshape(source,[-1])
    template_flatten = tf.reshape(template,[-1])
    #convert type to int 64
    source_flatten = tf.constant(source_flatten ,dtype =tf.int64)
    template_flatten = tf.constant(template_flatten ,dtype =tf.int64)

    src_values, src_unique_indices, src_counts = tf.unique_with_counts(source_flatten)
    tmpl_values, tmpl_counts = tf.unique_with_counts(template_flatten)

    # change to tensor 
    source_size = tf.size(source_flatten)
    template_size = tf.size(template_flatten)
    # calculate normalized quantiles for each array
    src_quantiles = tf.cumsum(src_counts) / source_size
    tmpl_quantiles = tf.cumsum(tmpl_counts) / template_size

    interp_a_values = tf.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)

def match_histograms(image, reference, *, multichannel=False):
    """Adjust an image so that its cumulative histogram matches that of another.
    The adjustment is applied separately for each channel.
    Parameters
    ----------
    image : ndarray
        Input image. Can be gray-scale or in color.
    reference : ndarray
        Image to match histogram of. Must have the same number of channels as
        image.
    multichannel : bool, optional
        Apply the matching separately for each channel.
    Returns
    -------
    matched : ndarray
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
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')

    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and '
                             'reference image must match!')

        matched = tf.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel],
                                                    reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)

    return matched
