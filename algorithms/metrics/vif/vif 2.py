# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
# Last update: 30/09/2019
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Remove Warnings

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR) # Remove Warnings

def normalized_gaussian_kernel(size, mean, std):
    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.cast(tf.range(-size, size+1), tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    gauss_kernel = tf.expand_dims(tf.expand_dims(gauss_kernel, 2), 3)
    
    return gauss_kernel / tf.reduce_sum(gauss_kernel)

def conv_padding_nearest(input, kernel):
    s = kernel.get_shape().as_list()[0]
    inputbig = nearest_padding(input, [0, s//2, s//2, 0], [0, (s-1)//2, (s-1)//2, 0])
    return tf.nn.convolution(inputbig, kernel, padding="VALID")

def nearest_padding(image, padding_left=[0,1,1,0], padding_right=None):
    if padding_right == None:
        padding_right = padding_left
    for i in range(0, 4):
        for j in range(0, padding_left[i]):
            image = tf.pad(image, [[k==i, 0] for k in range(0, 4)], "SYMMETRIC")
        for j in range(0, padding_right[i]):
            image = tf.pad(image, [[0, k==i] for k in range(0, 4)], "SYMMETRIC")
    return image

def conv_padding_symmetric(input, kernel):
    s = kernel.get_shape().as_list()[0]
    inputbig = tf.pad(input, [[0, 0], [s//2, (s-1)//2], [s//2, (s-1)//2], [0, 0]], "SYMMETRIC")
    return tf.nn.convolution(inputbig, kernel, padding="VALID")

def pbvif(ref, query_tab, max_scale=4, var_noise=2.0, mode="nearest"):
    """
    Computes the Pixel-Based Visual Information Fidelity (PB-VIF) using Tensorflow

    pbvif(ref, query_tab, max_scale=4, sigma_noise=2.0, mode="nearest")

    Parameters
    ----------
    ref       : 2-dimension grayscaled image reference.
    query_tab : list containing the 2-dimension grayscaled images to be compared with.
    max_scale : Number of subbands to extract information (Default: 4)
    var_noise : Variance of additive noise (HVS model parameter, Default: 2.0)
    mode      : mode used for padding convolutions (Default: "nearest")
        - "nearest"   : the input is extended by replicating the last pixel
        - "symmetric" : the input is extended by reflecting about the edge of the last pixel
        - "constant"  : the input is extended by filling all values beyond the edge with zeros

    Return
    ----------
    Pixel-Based Visual Information Fidelity (float between 0 and 1)
    """
    # Get function with right mode
    if mode == "nearest":
        conv = conv_padding_nearest
    elif mode == "symmetric":
        conv = conv_padding_symmetric
    elif mode == "constant":
        conv = lambda input, kernel: tf.nn.convolution(input, kernel, padding="SAME")
    else:
        raise NameError('Unknown mode: must be "nearest", "symmetric" or "constant"')
    
    # Variables
    ref = tf.Variable(ref, tf.float32)
    ref = tf.expand_dims(tf.expand_dims(ref, 0), 3)
    query = tf.placeholder(tf.float32)
    query_ = tf.expand_dims(tf.expand_dims(query, 0), 3)
    
    total_i_ref = tf.Variable(0.0, tf.float32)
    total_i_query = tf.Variable(0.0, tf.float32)
    for scale in range(0, max_scale):
        # Compute G field
        g_filter = normalized_gaussian_kernel(2**scale, 0.0, tf.cast(2**(scale+1)+1, tf.float32)/5.0)

        # Scale if necessary
        if scale < max_scale-1:
            ref2 = conv(ref, g_filter)[:, ::2, ::2, :]
            query2 = conv(query_, g_filter)[:, ::2, ::2, :]
        else:
            ref2 = ref
            query2 = query_

        # Compute expected matrices
        E_C = conv(ref2, g_filter)
        E_D = conv(query2, g_filter)

        # Compute covariance matrices
        cov_C_C = conv(tf.multiply(ref2, ref2), g_filter) - E_C*E_C
        cov_D_D = conv(tf.multiply(query2, query2), g_filter) - E_D*E_D
        cov_C_D = conv(tf.multiply(ref2, query2), g_filter) - E_C*E_D

        # Compute g_l and var_v_l
        g_l = cov_C_D/cov_C_C
        var_v_l = cov_D_D - g_l*cov_C_D

        # Compute information extracted by the brain
        i_ref = tf.math.log(1.0 + cov_C_C/var_noise)
        i_query = tf.math.log(1.0 + g_l*g_l*cov_C_C/(var_v_l+var_noise))

        # Add to total
        total_i_ref += tf.reduce_sum(i_ref)
        total_i_query += tf.reduce_sum(i_query)

    vif = tf.divide(total_i_query, total_i_ref)

    # Begin session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    vif_ = [sess.run(vif, feed_dict={query: query_tab[i]}) for i in range(0, len(query_tab))]
    
    sess.close()
    
    return vif_
