import numpy as np
import tensorflow as tf
from skimage.exposure import match_histograms
from skimage import data

def _match_cumulative_cdf(source, template):
    """
    Return modified source array so that the cumulative density function of
    its values matches the cumulative density function of the template.
    """
    sess = tf.InteractiveSession()
    source = tf.reshape(source, [-1])
    template = tf.reshape(template, [-1])
    src_values, _ = tf.unique(source)
    src_values = tf.sort(src_values)
    tf.global_variables_initializer().run()
    src_indices, src_counts = find_unique_indices(src_values, source)

    tmpl_values, _ = tf.unique(template)
    tmpl_values = tf.sort(tmpl_values)
    tmpl_indices, tmpl_counts = find_unique_indices(tmpl_values, template)
    # calculate normalized quantiles for each array
    tf.global_variables_initializer().run()
    src_quantiles = tf.divide(tf.cumsum(src_counts), tf.size(source))
    tmpl_quantiles = tf.divide(tf.cumsum(tmpl_counts), tf.size(template))
    interp_a_values = interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_indices.eval()].reshape(source.eval().shape)

def find_unique_indices(values, source):
    #values = values.eval()
    #source = source.eval()
    val_length = values.eval().shape[0]
    src_length = source.eval().shape[0]
    unique_indices = []
    unique_counts = np.zeros(val_length,dtype=np.int32)
    for i in range(0, src_length):
        for k in range(0, val_length):
            if source.eval()[i] == values.eval()[k]:
                unique_indices.append(k)
                unique_counts[k] = unique_counts[k] + 1
                break
    return tf.Variable(unique_indices), tf.Variable(unique_counts)


def interp(x, xp, fp):
    x = tf.cast(x, tf.float64)
    xp = tf.cast(xp, tf.float64)
    fp = tf.cast(fp, tf.float64)
    rise_vec = fp[1:] - fp[0:-1]
    run_vec = xp[1:] - xp[0:-1]
    grad_vec = tf.divide(rise_vec, run_vec)
    b = fp[0:-1] - tf.multiply(grad_vec, xp[0:-1])
    len_xp = xp.get_shape().as_list()[0]
    result = []
    x = x.eval()
    if x.size == 1:
        if x <= xp.eval()[0]:
            result.append(xp.eval()[0])
        elif x >= xp.eval()[-1]:
            result.append(xp.eval()[-1])
        else:
            for i in range(0, xp.eval().shape[0]-1):
                if x >= xp.eval()[i] and x<xp.eval()[i+1]:
                    result.append(x * grad_vec.eval()[i] + b.eval()[i])
                    break
    else:
        for value in x:
            if value <= xp.eval()[0]:
                result.append(xp.eval()[0])
            elif x >= xp.eval()[-1]:
                result.append(xp.eval()[-1])
            else:
                for i in range(0, xp.eval().shape[0] - 1):
                    if value >= xp.eval()[i] and value < xp.eval()[i + 1]:
                        result.append(value * grad_vec.eval()[i] + b.eval()[i])
                        break
    return np.array(result)

def my_match_histograms(image, reference, multichannel=False):
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
    image_shape = image.shape
    image_dtype = image.dtype
    ref_shape = reference.shape
    image = tf.Variable(image, dtype=tf.float64)
    reference = tf.Variable(reference, dtype=tf.float64)

    #if tf.rank(image).eval() != tf.rank(reference).eval():
    #    raise ValueError('Image and reference must have the same number of channels.')

    if multichannel:
        #if image_shape[-1] != ref_shape[-1]:
        #    raise ValueError('Number of channels in the input image and reference '
        #                     'image must match!')
        matched = np.empty(image_shape, dtype=image_dtype)
        for channel in range(image.get_shape().as_list()[-1]):
            matched_channel = _match_cumulative_cdf(image[..., channel], reference[..., channel])
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference)
    return matched


reference = data.coffee()
image = data.chelsea()

matched = match_histograms(image, reference, multichannel=True)
matched_tf = my_match_histograms(image, reference, multichannel=True)