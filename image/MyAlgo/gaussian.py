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
    
    # testing values
    start_pt = -10
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
    
    summm = 0
    #normalize
    for i in matrix:
        if i < 3:
            print("Inside for loop, i: ", i)
    # Printing
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    value_range = sess.run(value_range)
    values= sess.run(values)
    matrix = sess.run(matrix)
    print(values.shape)
    print(matrix.shape)
    return

# test run
m = 10
s = 2
make_gaussian_kernel(m, s)