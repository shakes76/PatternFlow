import tensorflow as tf
from scipy.signal import fftconvolve, convolve

def richardson_lucy(image, psf, iterations=50, clip=True):
    # Check for Complexity as in original original skimage function
    regular_complexity = tf.cast(tf.math.reduce_prod(image.shape + psf.shape), tf.float32)
    fft_complexity = tf.math.reduce_sum([tf.cast(n, tf.float32)*tf.math.log(tf.cast(n, tf.float32)) for n in image.shape + psf.shape])
    ratio = 40.032 * fft_complexity / regular_complexity

    # Cast tensors to float32 so they can be properly convolved
    image = tf.dtypes.cast(image, tf.float32)
    psf = tf.dtypes.cast(psf, tf.float32)
    im_deconv = tf.dtypes.cast(tf.fill(image.shape, 0.5), tf.float32)
    psf_mirror = psf[::-1, ::-1]

    if ratio <= 1 or len(image.shape) > 2:
        convolve_method = fftconvolve
    else:
        convolve_method = convolve

    # Calculates the convolution according to the number of times specified in the args
    for _ in range(iterations):
        relative_blur = image / convolve_method(im_deconv, psf, 'same')
        im_deconv *= convolve_method(relative_blur, psf_mirror, 'same')

    if clip:
        #Clips any values above 1 to 1 and below -1 to -1
        im_deconv = tf.clip_by_value(im_deconv, 1, -1)

    return im_deconv
