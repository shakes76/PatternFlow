#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:35:11 2019

@author: kajajuel
"""

import tensorflow as tf

def make_gaussian_kernel_two(mean, std, size):
    """
    Returns a 2D Gaussian kernel
    """
    #create normal distribution
    mean = tf.to_float(mean)
    std = tf.to_float(std)
    dist = tf.distributions.Normal(mean, std)
    
    start_pt = -size
    stop_pt = abs(start_pt) +1
    leng = abs(start_pt) + stop_pt
    value_range = tf.linspace(tf.to_float(start_pt), tf.to_float(stop_pt-1), leng, name="linspace")

    # find values by probability density function
    values = dist.prob(value_range)
    
    # make it 2D -> 2D[i,j] = values[i]*values[j]
    matrix = tf.einsum('i,j -> ij', values, values )
    
    # normalizing
    sum_of_matrix = tf.math.reduce_sum(matrix, axis=None, keepdims=False, name=None)
    gaussian_kernel = matrix/sum_of_matrix
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    gaussian_kernel = sess.run(gaussian_kernel)
    
    return gaussian_kernel


def convolve_two(img, kernel):
    """
    img = tensor
    kernel = 2D
    Returns image convolved with a gaussian kernel.
    """
    print("Running convolve")

    strides = [1,1,1,1] #list of ints
    
    # Operation 
    convolved = tf.nn.conv2d(img, kernel, strides = strides, padding = 'SAME')
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    convolved_op = sess.run(convolved)

    print("Done convolving")
    return convolved_op





