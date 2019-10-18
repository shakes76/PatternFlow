import tensorflow as tf 

def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    src_values, src_unique_indices, src_counts = tf.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = tf.unique(template.ravel(), return_counts=True)

    # calculate normalized quantiles for each array
    src_quantiles = tf.cumsum(src_counts) / source.size
    tmpl_quantiles = tf.cumsum(tmpl_counts) / template.size

    interp_a_values = tf.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)