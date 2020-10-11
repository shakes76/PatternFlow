"""
COMP3710 Assignment 3

Author: James Copperthwaite

Contrast Limited Adaptive Histogram Equalization (CLAHE).

"""


import tensorflow as tf
import numpy as np


def histogram(image, nbins=256, normalize=True):
    sess = tf.InteractiveSession()
    
    img = tf.constant(image.astype(np.float32))
    hist = tf.Variable(np.zeros(nbins).astype(np.int64))
    out = tf.Variable(img)

    tf.global_variables_initializer().run() #init variables

    img = tf.reshape(img, [-1]) 

    current_min = tf.dtypes.cast(tf.reduce_min(img), tf.float32)
    current_max = tf.dtypes.cast(tf.reduce_max(img), tf.float32)
    
    bins = tf.linspace(current_min, current_max, nbins+1)
    
    y, idx, count = tf.unique_with_counts(img)

    for i in range(nbins):
        if i==nbins-1:
            # if last bin cut off at less than or equal to the bin limit
            mask = (img <= bins[i+1]) 
            lim = tf.boolean_mask(img, mask)

        else: # cut off bin at less than upper bound 
            mask = (img < bins[i+1]) 
            lim = tf.boolean_mask(img, mask)
        mask = (lim >= bins[i]) # all bins are greater than or equal to lower bound
        lim = tf.boolean_mask(img, mask)

        vals = tf.dtypes.cast(mask, tf.int32)
        count = tf.count_nonzero(vals)

        idx = tf.dtypes.cast(tf.one_hot(i, nbins), tf.int64)  # output: [3 x 3]
        idx = tf.math.scalar_mul(count, idx)
        hist = hist + idx

    mids = (bins[:-1] + bins[1:]) / 2
    h = hist.eval()
    bin_centers = mids.eval()
    sess.close()
    return h, bin_centers


    


