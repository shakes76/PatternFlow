import tensorflow as tf

tf.compat.v1.enable_eager_execution()


def _interpolate( dx_T, dy_T, x, name='interpolate' ):
    
    
    with tf.compat.v1.variable_scope(name):

        with tf.compat.v1.variable_scope('neighbors'):

            delVals = tf.subtract(dx_T, x)
            ind_1   = tf.argmax(tf.sign( delVals ))
            ind_0   = ind_1 - 1

        with tf.compat.v1.variable_scope('calculation'):

            value   = tf.cond( x[0] <= dx_T[0], 
                              lambda : dy_T[:1], 
                              lambda : tf.cond( 
                                     x[0] >= dx_T[-1], 
                                     lambda : dy_T[-1:],
                                     lambda : (dy_T[ind_0] +                
                                               (dy_T[ind_1] - dy_T[ind_0])  
                                               *(x-dx_T[ind_0])/            
                                               (dx_T[ind_1]-dx_T[ind_0]))
                             ))

        result = tf.multiply(value[0], 1, name='y')

    return result

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
    #interpolate
    interp_a_values = []
    for i in src_quantiles.numpy():
        interp_a_values.append(_interpolate(tmpl_quantiles, tmpl_values, tf.constant([i])))
    interp_a_values = tf.convert_to_tensor(interp_a_values).numpy()
    guodu = interp_a_values[src_unique_indices]
    #convert_to_tensor
    guodu = tf.convert_to_tensor(guodu)
    result = tf.reshape(guodu,tf.shape(source))
    return result

    
def match_histograms(image, reference, multichannel=False):
     
    if tf.rank(image).numpy() != tf.rank(reference).numpy():
        raise ValueError('Image and reference must have the same number of channels.')
    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and reference '
                             'image must match!')

        
        matched_channel = []
    
        for channel in range(image.shape[-1]):
            matched_channel.append(_match_cumulative_cdf(image[..., channel], reference[..., channel]))
            
        matched = tf.stack([matched_channel[0],matched_channel[1],matched_channel[2]], axis=2)

        matched = matched/255.
    
    else:
        matched = _match_cumulative_cdf(image, reference)
        matched = matched/255.
    return matched
