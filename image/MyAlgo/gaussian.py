# -*- coding: utf-8 -*-


# steps
    # make 2D gaussian kernel
    # convolve kernel with image
    # done!
    # add multiple dim if time
import tensorflow as tf

def make_gaussian_kernel(mean, std):
    """Returns a 2D Gaussian kernel"""
    #create normal distribution
    mean = float(mean)
    std = float(std)
    dist = tf.distributions.Normal(mean, std)
    
    #testing values
    start_pt = -10
    stop_pt = abs(start_pt) +1
    leng = abs(start_pt) + stop_pt
    value_range = tf.linspace(float(start_pt), float(stop_pt-1), leng, name="linspace")
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    value_range = sess.run(value_range)
    
    #print("value_range  : ", value_range)
    
    values = dist.prob(value_range)
    
    # make it 2D -> 2D[i,j] = values[i]*values[j]
    # x=tf.tensordot(A_tf, B_tf,axes = [[1], [0]])
    # x.get_shape()
    
    x = tf.tensordot(values, values, axes = [[1], [0]])
    
    print(values.shape)
    return

# test run
m = 10
s = 2
make_gaussian_kernel(m, s)