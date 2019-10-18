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
    mean = tf.to_float(mean)
    std = tf.to_float(std)
    dist = tf.distributions.Normal(mean, std)
    
    start_pt = -size
    stop_pt = abs(start_pt) +1
    leng = abs(start_pt) + stop_pt
    value_range = tf.linspace(tf.to_float(start_pt), tf.to_float(stop_pt-1), leng, name="linspace")
    
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
#    sess = tf.Session()
#    sess.run(tf.global_variables_initializer())
#    value_range = sess.run(value_range)
#    values= sess.run(values)
#    matrix = sess.run(matrix)
#    gaussian_kernel = sess.run(gaussian_kernel)
    
    #print(values.shape)
    #print(gaussian_kernel)
    print("Gaussian kernel made")
    return gaussian_kernel

# test run
#m = 10
#s = 2
#size = 10
#gaussian = make_gaussian_kernel(m, s, size)


## expand its dimensionality to fit into conv2d
#tensor_expand = tf.expand_dims(tensor, 0)
#tensor_expand = tf.expand_dims(tensor_expand, 0)
#tensor_expand = tf.expand_dims(tensor_expand, -1)
#print(tensor_expand.get_shape()) # => (1, 1, 100, 1)
#
## do the same in one line with reshape
#tensor_reshape = tf.reshape(tensor, [1, 1, tensor.get_shape().as_list()[0],1])
#print(tensor_reshape.get_shape()) # => (1, 1, 100, 1)

def convolve(img, kernel, rgb = True):
    """
    img = tensor
    Returns image convolved with a gaussian kernel.
    """
    print("Running convolve")
    #with tf.Graph().as_default():      

    #if rgb:
        #num_maps = 3
    #else:
        #num_maps = 1

    
    strides = [1,1,1,1] #list of ints
    print("so far, so good")
    #x = tf.placeholder('float32', [None, None, None, num_maps])
    #w = tf.get_variable('w', initializer = tf.to_float(kernel))
    
    
    #print("x.shape: ", x.shape)
    #print("w.shape: ", w.shape)
    
    # Operation 
    convolved = tf.nn.conv2d(img, kernel, strides = strides, padding = 'SAME')
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    convolved_op = sess.run(convolved)
#    x = tf.placeholder('float32', [None, None, None, num_maps])
#    w = tf.get_variable('w', initializer = tf.to_float(kernel))
#    
#    
#    print("x.shape: ", x.shape)
#    print("w.shape: ", w.shape)
#    
#    # Operation 
#    convolved = tf.nn.conv2d(x, w, strides = strides, padding = 'SAME')
#    
#    sess = tf.Session()
#    sess.run(tf.global_variables_initializer())
#    convolved_op = sess.run(convolved, feed_dict={x: img})
   
  
    
    
    
    print("Done convolving")
    return conv_op



        
















