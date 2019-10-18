#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:35:11 2019

@author: kajajuel
"""

import tensorflow as tf

def make_gaussian_kernel_two(mean, sd, size):
    """
    Returns a 2D Gaussian kernel
    """
    #create normal distribution
    mean = tf.to_float(mean)
    sd = tf.to_float(sd)
    size = tf.to_float(size)
    dist = tf.distributions.Normal(mean, sd)
    
    start_pt = - size
    stop_pt = abs(start_pt)
    length = tf.to_int32(abs(start_pt) + stop_pt + 1)
    value_range = tf.linspace(start_pt, stop_pt, length, name="linspace")

    # find values by probability density function
    values = dist.prob(value_range)
    
    # make it 2D -> 2D[i,j] = values[i]*values[j]
    matrix = tf.einsum('i,j -> ij', values, values )
    
    # normalizing
    sum_of_matrix = tf.reduce_sum(matrix)
    gaussian_kernel = matrix/sum_of_matrix
    return gaussian_kernel

def convolve_two(img, kernel):
    """
    img = 4D tensor
    kernel = 4D tensor
    Returns image convolved with a gaussian kernel.
    """
    print("Running convolve")
    strides = [1,1,1,1] #list of ints
    
    # Operation 
    convolved = tf.nn.conv2d(img, kernel, strides = strides, padding = 'SAME')

    print("Done convolving")
    return convolved





