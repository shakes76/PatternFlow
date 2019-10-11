import tensorflow as tf


def _circulant2_dx(xs, dir):
    stack = [xs[:, dir:], xs[:, :dir]] if dir > 0 else [xs[:, dir:], xs[:, :dir]]
    shift = tf.concat(stack, axis=1)
    return shift - xs


def _circulant2_dy(xs, dir):
    stack = [xs[dir:, :], xs[:dir, :]] if dir > 0 else [xs[dir:, :], xs[:dir, :]]
    shift = tf.concat(stack, axis=0)
    return shift - xs


def _apply_to_channel(image, function):
    # Apply the function to each channel and then re-stack and return
    channels = tf.unstack(image, axis=-1)
    results = [function(channel) for channel in channels]
    return tf.stack(results, -1)


def _fft_channel(image):
    # Ensure the input is complex
    image = tf.cast(image, tf.complex64)

    return _apply_to_channel(image, tf.signal.fft2d)


def _ifft_channel(image):
    # Ensure the input is complex
    image = tf.cast(image, tf.complex64)

    return _apply_to_channel(image, tf.signal.ifft2d)


def l0_smoothing(image, lmd=0.01, beta_max=10000, beta_rate=2., max_iterations=30):
    # Ensure that the image is a Tensor
    image = tf.convert_to_tensor(image, tf.float32)
    image = tf.complex(image, tf.zeros_like(image))
    assert len(image.shape) == 2 or len(image.shape) == 3, f'Image should be either rank 2 or rank 3 not rank {len(image.shape)}'

    # If the image has no channel dimension so add one
    if len(image.shape) == 2:
        image = tf.expand_dims(image, -1)

    rows, cols, channels = image.shape

    #Ny = rows, Nx = cols, D = channels
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
    beta = lmd * 2.0
    hp, vp = tf.zeros_like(image), tf.zeros_like(image)
    for i in range(max_iterations):
        # With S, solve for hp and vp
        hp, vp = _circulant2_dx(S, 1), _circulant2_dy(S, 1)
        mask = tf.greater_equal(tf.reduce_sum(hp ** 2. + vp ** 2., axis=2), lmd/beta * tf.ones([rows, cols]))

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
        if beta > beta_max: break

    return S