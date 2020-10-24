from typing import Callable

import tensorflow as tf


def _apply_to_channel(image: tf.Tensor, function: Callable[[tf.Tensor], tf.Tensor]) -> tf.Tensor:
    """Apply a function to each channel in an image (assumes channel last)"""
    channels = tf.unstack(image, axis=-1)
    results = [function(channel) for channel in channels]
    return tf.stack(results, -1)


def _fft_channel(image: tf.Tensor) -> tf.Tensor:
    """Returns the fast fourier transform of the given image applied individually to each channel"""
    # Ensure the input is complex
    image = tf.cast(image, tf.complex64)
    return _apply_to_channel(image, tf.signal.fft2d)


def _ifft_channel(image: tf.Tensor) -> tf.Tensor:
    """Returns the inverse fast fourier transform of the given image applied individually to each channel"""
    # Ensure the input is complex
    image = tf.cast(image, tf.complex64)
    return _apply_to_channel(image, tf.signal.ifft2d)


def _circulant2_dx(xs: tf.Tensor, dir: int) -> tf.Tensor:
    """Get the next circulant matrix for the dx in the specified direction"""
    stack = [xs[:, dir:], xs[:, :dir]] if dir > 0 else [xs[:, dir:], xs[:, :dir]]
    shift = tf.concat(stack, axis=1)
    return shift - xs


def _circulant2_dy(xs: tf.Tensor, dir: int) -> tf.Tensor:
    """Get the next circulant matrix for the dy in the specified direction"""
    stack = [xs[dir:, :], xs[:dir, :]] if dir > 0 else [xs[dir:, :], xs[:dir, :]]
    shift = tf.concat(stack, axis=0)
    return shift - xs


def l0_gradient_smoothing(image, smoothing_factor: float=0.01, beta_max: int=10000, beta_rate: float=2., max_iterations: int=30) -> tf.Tensor:
    """
    Performs l0 gradient smoothing on the given input data.
    This is essentially a tensorflow port of the code found here: https://github.com/t-suzuki/l0_gradient_minimization_test

    :param image: Input image or data where the last dimension corresponds to the channels.
                  An arbitrary number of channels is supported.
                  Input should be in the range [0, 1].
    :param smoothing_factor: Increasing this value will make the result smoother
    :param beta_max: Termination parameter
    :param beta_rate: The rate at which to grow beta
    :param max_iterations: The maximum number of iterations
    :return: Smoothed result
    """
    # Ensure that the image is a Tensor
    image = tf.convert_to_tensor(image, tf.float32)
    image = tf.complex(image, tf.zeros_like(image))
    assert len(image.shape) == 2 or len(image.shape) == 3, \
        f'Image should be either rank 2 or rank 3 not rank {len(image.shape)}'

    # If the image has no channel dimension so add one
    if len(image.shape) == 2:
        image = tf.expand_dims(image, -1)

    rows, cols, channels = image.shape

    # Create the optical transfer function
    dx, dy = tf.zeros((rows, cols), dtype=tf.complex64), tf.zeros((rows, cols), dtype=tf.complex64)
    dx = tf.tensor_scatter_nd_update(dx, [[rows//2, cols//2 - 1], [rows//2, cols//2]], [-1, 1])
    dy = tf.tensor_scatter_nd_update(dy, [[rows//2 - 1, cols//2], [rows//2, cols//2]], [-1, 1])

    F_denom = tf.abs(tf.signal.fft2d(dx)) ** 2. + tf.abs(tf.signal.fft2d(dy)) ** 2.
    if channels > 1:
        F_denom = tf.stack([F_denom]*channels, -1)

    # Take the fourier transform of the original image
    image_fourier = _fft_channel(image)

    # Start the optimisation process
    S = tf.math.real(image)
    beta = smoothing_factor * 2.0
    for i in range(max_iterations):
        # With S, solve for hp and vp
        hp, vp = _circulant2_dx(S, 1), _circulant2_dy(S, 1)
        mask = tf.greater_equal(tf.reduce_sum(hp ** 2. + vp ** 2., axis=2), smoothing_factor / beta * tf.ones([rows, cols]))

        # Set the values in hp and vp to 0 according to the mask
        mask_values = tf.cast(mask, tf.float32)
        mask_values = tf.stack([mask_values]*channels, -1)
        hp = tf.multiply(hp, mask_values)
        vp = tf.multiply(vp, mask_values)

        # With hp and vp, solve for S
        hv = _circulant2_dx(hp, -1) + _circulant2_dy(vp, -1)

        num = (image_fourier + (beta * _fft_channel(hv)))
        den = tf.cast(1.0 + beta * F_denom, tf.complex64)
        S = tf.math.real(_ifft_channel(num / den))

        # Iteration step
        beta *= beta_rate
        if beta > beta_max:
            break

    return S
