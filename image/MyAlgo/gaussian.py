# -*- coding: utf-8 -*-


# steps
    # make 2D gaussian kernel
    # convolve kernel with image
    # done! haha
    # add multiple dim if time
    
import tensorflow as tf
#from tensorflow import keras
#from keras.layers import Conv2D

def make_gaussian_kernel(mean, std, size):
    """
    Returns a 2D Gaussian kernel
    """
    print("Running make_gaussian_kernel")
    #create normal distribution
    mean = float(mean)
    std = float(std)
    dist = tf.distributions.Normal(mean, std)
    
    start_pt = -size
    stop_pt = abs(start_pt) +1
    leng = abs(start_pt) + stop_pt
    value_range = tf.linspace(float(start_pt), float(stop_pt-1), leng, name="linspace")
    
    # print("value_range  : ", value_range)
    # find values by probability density function
    values = dist.prob(value_range)
    
    
    
    # make it 2D -> 2D[i,j] = values[i]*values[j]
    # Outer product
    # syntax: einsum('i,j->ij', u, v)  # output[i,j] = u[i]*v[j]
    matrix = tf.einsum('i,j -> ij', values, values )
    
    # normalizing
    sum_of_matrix = tf.math.reduce_sum(matrix, axis=None, keepdims=False, name=None)
    gaussian_kernel = matrix/sum_of_matrix
    
    # Printing
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    value_range = sess.run(value_range)
    values= sess.run(values)
    matrix = sess.run(matrix)
    gaussian_kernel = sess.run(gaussian_kernel)
    
    #print(values.shape)
    #print(gaussian_kernel)
    print("Gaussian kernel made")
    return gaussian_kernel

# test run
m = 10
s = 2
size = 10
gaussian = make_gaussian_kernel(m, s, size)


def convolve(img, kernel):
    """
    Returns image convolved with a gaussian kernel.
    """
#   x = tf.expand_dims(w, 2)
#   x = tf.expand_dims(x, 3)
    kernel_3D = tf.expand_dims(kernel, 2)
    kernel_4D = tf.expand_dims(kernel_3D, 3)
    strides = [1,1,1,1] #list of ints
    #tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None, name=None)
#   filter must be A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]!!
    
    convolved = tf.nn.conv2d(img, kernel_4D, strides, padding = 'SAME', data_format='NHWC', dilations=None, name=None)
    return convolved




















