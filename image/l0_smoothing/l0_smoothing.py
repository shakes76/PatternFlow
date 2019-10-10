import tensorflow as tf

def l0_smoothing(image, lmd=0.01, beta_max=10000, beta_rate=2., max_iterations=30):
    # Ensure that the image is a Tensor
    image = tf.convert_to_tensor(image, tf.float32)
    image = tf.complex(image, tf.zeros_like(image))
    assert len(image.shape) == 2 or len(image.shape) == 3, f'Image should be either rank 2 or rank 3 not rank {len(image.shape)}'

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

    # Get the fourier transform of the image
    image_fourier = tf.signal.fft2d(tf.squeeze(image))

    return F_denom


if __name__ == '__main__':
    a = l0_smoothing(tf.ones([6, 6, 3]))
    print(a)