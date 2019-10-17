# -*- coding: utf-8 -*-

"""
Created on Wed Oct 16 09:39:38 2019

@author: kajajuel
"""

import tensorflow as tf
# ----------------------------------------------------------------------
# EXAMPLE USE KERAS
#import numpy as np
#
#model = tf.keras.models.Sequential()
#model.add(tf.keras.layers.Dense(1, input_dim = 1))
#
#model.compile(loss = 'mean_squared_error', optimizer = 'sgd')
#x_samples = np.array([-1, 0, 1, 2, 3, 4])
#y_samples = np.array([-3, -1, 1, 3, 5, 7])
#
#model.fit(x_samples, y_samples, epochs = 500)
#to_predict = np.array([10, 11, 12, 13])
#
#print(model.predict(to_predict))


# Both tf.tensordot() and tf.einsum() are syntactic sugar that wrap one 
# or more invocations of tf.matmul() (although in some special cases
# tf.einsum() can reduce to the simpler elementwise tf.multiply())
# ---------------------------------------------------------------------
# EXAMPLE EINSUM
#u = tf.ones(4, tf.int32)
#w = tf.einsum('i,j->ij', u, u)
#tf.math.reduce_sum(
#    input_tensor,
#    axis=None,
#    keepdims=False,
#    name=None
#)
#
#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#
#u = sess.run(u)
#w = sess.run(w)
#
#print("u.shape: ", u.shape)
#print("w.shape: ", w.shape)
# ---------------------------------------------------------------------
# Create model
# From lab 2
#from keras.layers import Dense, Activation, Conv2D,FLatten
#model = Sequential()
#
## Adding model layers
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(50,37,1)))
#model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))

#https://mourafiq.com/2016/08/10/playing-with-convolutions-in-tensorflow.html
#def convolve(img, kernel, strides=[1, 3, 3, 1], pooling=[1, 3, 3, 1], padding='SAME', rgb=True):
# 2     with tf.Graph().as_default():
# 3         num_maps = 3
# 4         if not rgb:
# 5             num_maps = 1  # set number of maps to 1
# 6             img = img.convert('L', (0.2989, 0.5870, 0.1140, 0))  # convert to gray scale
# 7 
# 8 
# 9         # reshape image to have a leading 1 dimension
#10         img = numpy.asarray(img, dtype='float32') / 256.
#11         img_shape = img.shape
#12         img_reshaped = img.reshape(1, img_shape[0], img_shape[1], num_maps)
#13 
#14         x = tf.placeholder('float32', [1, None, None, num_maps])
#15         w = tf.get_variable('w', initializer=tf.to_float(kernel))
#16 
#17         # operations
#18         conv = tf.nn.conv2d(x, w, strides=strides, padding=padding)
#19         sig = tf.sigmoid(conv)
#20         max_pool = tf.nn.max_pool(sig, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding=padding)
#21         avg_pool = tf.nn.avg_pool(sig, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding=padding)
#22 
#23         init = tf.initialize_all_variables()
#24         with tf.Session() as session:
#25             session.run(init)
#26             conv_op, sigmoid_op, avg_pool_op, max_pool_op = session.run([conv, sig, avg_pool, max_pool],
#27                                                                         feed_dict={x: img_reshaped})
#28 
#29         show_shapes(img, conv_op, sigmoid_op, avg_pool_op, max_pool_op)
#30         if rgb:
#31             show_image_ops_rgb(img, conv_op, sigmoid_op, avg_pool_op, max_pool_op)
#32         else:
#33             show_image_ops_gray(img, conv_op, sigmoid_op, avg_pool_op, max_pool_op)



#tf.nn.conv2d(input, filters, strides, padding, data_format='NHWC', dilations=None, name=None)

# filter must be A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]!!

#input: A Tensor. Must be one of the following types: half, bfloat16, float32, float64. A 4-D tensor. The dimension order is interpreted according to the value of data_format, see below for details.
#filters: A Tensor. Must have the same type as input. A 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
#strides: An int or list of ints that has length 1, 2 or 4. The stride of the sliding window for each dimension of input. If a single value is given it is replicated in the H and W dimension. By default the N and C dimensions are set to 1. The dimension order is determined by the value of data_format, see below for details.
#padding: Either the string "SAME" or "VALID" indicating the type of padding algorithm to use, or a list indicating the explicit paddings at the start and end of each dimension. When explicit padding is used and data_format is "NHWC", this should be in the form [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]]. When explicit padding used and data_format is "NCHW", this should be in the form [[0, 0], [0, 0], [pad_top, pad_bottom], [pad_left, pad_right]].
#data_format: An optional string from: "NHWC", "NCHW". Defaults to "NHWC". Specify the data format of the input and output data. With the default format "NHWC", the data is stored in the order of: [batch, height, width, channels]. Alternatively, the format could be "NCHW", the data storage order of: [batch, channels, height, width].
#dilations: An int or list of ints that has length 1, 2 or 4, defaults to 1. The dilation factor for each dimension ofinput. If a single value is given it is replicated in the H and W dimension. By default the N and C dimensions are set to 1. If set to k > 1, there will be k-1 skipped cells between each filter element on that dimension. The dimension order is determined by the value of data_format, see above for details. Dilations in the batch and depth dimensions if a 4-d tensor must be 1.
#name: A name for the operation (optional).

u = tf.ones(10, tf.int32)
w = tf.einsum('i,j->ij', u, u)
v = w[:,1]
x = tf.expand_dims(w, 2)
x = tf.expand_dims(x, 3)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

u = sess.run(u)
v = sess.run(v)
w = sess.run(w)
x = sess.run(x)

print("w.shape: ", w.shape)
print("x.shape: ", x.shape)


# expand its dimensionality to fit into conv2d
tensor_expand = tf.expand_dims(tensor, 0)
tensor_expand = tf.expand_dims(tensor_expand, 0)
tensor_expand = tf.expand_dims(tensor_expand, -1)
print(tensor_expand.get_shape()) # => (1, 1, 100, 1)

# do the same in one line with reshape
tensor_reshape = tf.reshape(tensor, [1, 1, tensor.get_shape().as_list()[0],1])
print(tensor_reshape.get_shape()) # => (1, 1, 100, 1)




















