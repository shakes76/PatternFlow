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
