#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 10:35:11 2019

@author: kajajuel
"""

import tensorflow as tf

def make_gaussian_kernel(mean, sd, size):
    """
    Returns a 2D Gaussian kernel for convolution.
    Arguments:
        mean: the mean of the gaussian distribution
        sd:   the standard deviation of the gaussian distribution
        size: the size to define matrix
    """
    #create normal distribution
    mean = tf.to_float(mean)
    sd = tf.to_float(sd)
    size = tf.to_float(size)
    dist = tf.distributions.Normal(mean, sd)
    
    start_pt = - size
    stop_pt = abs(start_pt)
    length = tf.to_int32(abs(start_pt) + stop_pt + 1)
    value_range = tf.linspace(start_pt, stop_pt, length)

    # find values by probability density function
    values = dist.prob(value_range)
    
    # make it 2D -> 2D[i,j] = values[i]*values[j]
    matrix = tf.einsum('i,j -> ij', values, values)
    
    # normalizing
    sum_of_matrix = tf.reduce_sum(matrix)
    gaussian_kernel = matrix/sum_of_matrix
    return gaussian_kernel

def convolve(img, m, s, sz):
    """
    Convolves image img with a gaussian kernel specified by kernel. 
    Arguments:
        img:       an image represented by a 4D tensor
        m, s, sz:  the values specifying the gaussian distribution to be created,
                   m is the mean, s is the standerd deviation and sz is the size
    """
    
    kernel = make_gaussian_kernel(m, s, sz)
    kernel_4D = tf.reshape(kernel, [kernel.shape[0], kernel.shape[1], 1 ,1])
    strides = [1,1,1,1]
    
    # Operation 
    convolved = tf.nn.conv2d(img, kernel_4D, strides = strides, padding = 'SAME')

    return convolved





