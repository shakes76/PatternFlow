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

def convolve(img, kernel):
    """
    Returns image convolved with a gaussian kernel.
    """
    print("Running convolve")
    with tf.Graph().as_default():
    
        # normalise
        img = img / 255.0
      
        #reshape
        #print("kernel_4D.shape: ", kernel_4D.shape)
        #print("img_4D.shape: ", img_4D.shape)
        kernel_4D = tf.reshape(kernel, [kernel.shape[0], kernel.shape[1], 1 ,1])
        img_4D = tf.reshape(img, [img.shape[0], img.shape[1], img.shape[2], 1])
        
        strides = [1,1,1,1] #list of ints
        # tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None, name=None)
        # filter must be A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]!!
        
        x = tf.placeholder('float32')
        w = tf.get_variable('w', initializer = tf.to_float(kernel_4D))
        
        print("so far, so good")
  
    
        #convolved = tf.nn.conv2d(img_4D, kernel_4D, strides, padding = 'SAME', data_format='NHWC')
        convolved = tf.nn.conv2d(x, w, strides = strides, padding = 'SAME')
        #init = tf.initialize_all_variables()
        #with tf.Session() as sess:
        #    sess.run(init)
        #    conv_op = sess.run(convolved, feed_dict={x: img_4D})
    #    
    #    print("Done convolving")
#    
#    return conv_op



        
















