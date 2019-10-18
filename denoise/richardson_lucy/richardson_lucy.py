import tensorflow as tf
#import numpy as np
import scipy as sp

def richardson_lucy(image, psf, iterations=50, clip=True):
    # Check for Complexity as in original original skimage function
    regular_complexity = tf.cast(tf.math.reduce_prod(image.shape + psf.shape), tf.float32)
    fft_complexity = tf.math.reduce_sum([tf.cast(n, tf.float32)*tf.math.log(tf.cast(n, tf.float32)) for n in image.shape + psf.shape])
    ratio = 40.032 * fft_complexity / regular_complexity

    image = tf.dtypes.cast(image, tf.float32)
    psf = tf.dtypes.cast(psf, tf.float32)
    im_deconv = tf.dtypes.cast(tf.fill(image.shape, 0.5), tf.float32)
    psf_mirror = psf[::-1, ::-1]

    if ratio <= 1 or len(image.shape) > 2:
        convolve_method = sp.signal.fftconvolve
    else:
        convolve_method = sp.signal.convolve

    for _ in range(iterations):
        relative_blur = image / convolve_method(im_deconv, psf, 'same')
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')

    return im_deconv
