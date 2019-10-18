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

    src_values, src_unique_indices, src_counts = tf.unique(source_flatten)
    tmpl_values, tmpl_counts = tf.unique(template_flatten)

    # change to tensor 
    source_size = tf.size(source_flatten)
    template_size = tf.size(template_flatten)
    # calculate normalized quantiles for each array
    src_quantiles = tf.cumsum(src_counts) / source_size
    tmpl_quantiles = tf.cumsum(tmpl_counts) / template_size

    interp_a_values = tf.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)