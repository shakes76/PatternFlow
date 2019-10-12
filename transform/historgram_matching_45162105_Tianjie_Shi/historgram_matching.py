import tensorflow as tf

tf.compat.v1.enable_eager_execution()

def _match_cumulative_cdf(source, template):
    	source_flatten = tf.reshape(source,[-1])
    template_flatten = tf.reshape(template,[-1])
    src_values, src_unique_indices, src_counts = tf.unique_with_counts(source_flatten)
    tmpl_values,tmpl_unique_indices,tmpl_counts = tf.unique_with_counts(template_flatten)
    tmpl_values = tf.cast(tmpl_values,dtype=tf.float64)
    source_size = tf.size(source_flatten)
    template_size = tf.size(template_flatten)
    src_quantiles = tf.cumsum(src_counts) / source_size
    tmpl_quantiles = tf.cumsum(tmpl_counts) / template_size
