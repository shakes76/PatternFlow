# -*- coding: utf-8 -*-
# Author: Antoine DELPLACE
import tensorflow as tf

def gaussian_kernel(size, mean, std):
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

def pbvif(ref, query, max_scale=4, sig=2.0, mode="nearest"):
    """
    Computes the Pixel-Based Visual Information Fidelity (PB-VIF) using Tensorflow

    max_scale = number of subbands

    sig = sigma_n^2/lambda_k

    Note to myself: for convolutions: input=gauss_noise, kernel=image

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
    query = tf.Variable(query, tf.float32)
    query = tf.expand_dims(tf.expand_dims(query, 0), 3)
    
    total_i_ref = tf.Variable(0.0, tf.float32)
    total_i_query = tf.Variable(0.0, tf.float32)
    for scale in range(0, max_scale):
        # Compute mu and sigma
        gk = gaussian_kernel(2**scale, 0.0, tf.cast(2**(scale+1)+1, tf.float32)/5.0)

        if scale < max_scale-1:
            ref2 = conv(ref, gk)[:, ::2, ::2, :]
            query2 = conv(query, gk)[:, ::2, ::2, :]
        else:
            ref2 = ref
            query2 = query

        mu_ref = conv(ref2, gk)
        mu_query = conv(query2, gk)

        sigma_ref2 = conv(tf.multiply(ref2, ref2), gk) - mu_ref*mu_ref
        sigma_query2 = conv(tf.multiply(query2, query2), gk) - mu_query*mu_query
        sigma_ref_query = conv(tf.multiply(ref2, query2), gk) - mu_ref*mu_query

        g = sigma_ref_query/sigma_ref2
        s = sigma_query2 - g*sigma_ref_query

        # Compute information extracted by the brain
        i_ref = tf.math.log(1.0 + sigma_ref2/sig)
        i_query = tf.math.log(1.0 + g*g*sigma_ref2/(s+sig))

        # Add to total
        total_i_ref += tf.reduce_sum(i_ref)
        total_i_query += tf.reduce_sum(i_query)

    vif = tf.divide(total_i_query, total_i_ref)
        
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    vif_ = sess.run(vif)
    
    sess.close()
    
    return vif_
