import tensorflow as tf
#import numpy as np

def richardson_lucy(image, psf, iterations=50, clip=True):
    # Check for Complexity as in original original skimage function
    regular_complexity = tf.cast(tf.math.reduce_prod(image.shape + psf.shape), tf.float32)
    fft_complexity = tf.math.reduce_sum([tf.cast(n, tf.float32)*tf.math.log(tf.cast(n, tf.float32)) for n in image.shape + psf.shape])
    ratio = 40.032 * fft_complexity / regular_complexity

    return im_deconv
