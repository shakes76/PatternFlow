import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import math

from tensorflow.keras.preprocessing.image import apply_affine_transform as affine

def radon(image, theta=None, circle=True, *, preserve_range=None):
    """
    Calculates the radon transform of an image given specified
    projection angles.
    
    Parameters
    ----------
    image : array_like
        Input image. The rotation axis will be located in the pixel with
        indices ``(image.shape[0] // 2, image.shape[1] // 2)``.
    theta : array_like, optional
        Projection angles (in degrees). If `None`, the value is set to
        np.arange(180).
    circle : boolean, optional
        Assume image is zero outside the inscribed circle, making the
        width of each projection (the first dimension of the sinogram)
        equal to ``min(image.shape)``.
    preserve_range : bool, optional
        Whether to keep the original range of values. Otherwise, the input
        image is converted according to the conventions of `img_as_float`.
        Also see https://scikit-image.org/docs/dev/user_guide/data_types.html
    Returns
    -------
    radon_image : array
        Radon transform (sinogram).  The tomography rotation axis will lie
        at the pixel index ``radon_image.shape[0] // 2`` along the 0th
        dimension of ``radon_image``.
    """
    if image.dtype != tf.uint32:
        image = tf.cast(image, tf.uint8)
    
    if len(image.shape) > 2:
        image = tf.image.rgb_to_grayscale(image)
    
    if theta is None:
        theta = tf.range(180)
    
    if circle:
        img_shape = image.shape[:2]
        shape_min = tf.math.reduce_min(img_shape)
        radius = tf.math.floordiv(shape_min, 2)
        x, y = tf.meshgrid(range(0, img_shape[0]), range(0, img_shape[1]))
        center_x, center_y = tf.math.floordiv(img_shape, 2)
        dist = (x - center_x) ** 2 + (y - center_y) ** 2
        outside_reconstruction_circle = dist > radius ** 2
        if tf.math.reduce_any(image[outside_reconstruction_circle] != 0):
            print('Radon transform: image must be zero outside the '
                'reconstruction circle')
        # Crop image to make it square
        slices = tuple(
            slice(int(tf.math.ceil(excess / 2).numpy()),
                int(tf.math.ceil(excess / 2) + shape_min).numpy())
            if excess > 0 else
            slice(None)
            for excess in (img_shape - shape_min)
        )
        padded_image = image[slices]
    else:
        diagonal = tf.math.sqrt(2) * max(image.shape)
        pad = [int(tf.math.ceil(diagonal - s)) for s in image.shape]
        new_center = [tf.math.floordiv(s + p, 2) for s, p in zip(image.shape, pad)]
        old_center = [s // 2 for s in image.shape]
        pad_before = [nc - oc for oc, nc in zip(old_center, new_center)]
        pad_width = [(pb, p - pb) for pb, p in zip(pad_before, pad)]
        padded_image = tf.pad(image, pad_width, mode='constant',
            constant_values=0)
    # padded_image is always square
    if padded_image.shape[0] != padded_image.shape[1]:
        raise ValueError('padded_image must be a square')
    center = tf.cast(tf.math.floordiv(padded_image.shape[0], 2), tf.float32)
    #radon_image = tf.zeros((padded_image.shape[0], len(theta)))
    
    x, y = tf.meshgrid(tf.range(padded_image.shape[0], dtype=tf.int64), tf.range(padded_image.shape[1], dtype=tf.int64))
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])
    coords = tf.stack([x, y], 1)
    result = []
    diameter = tf.cast(tf.range(-radius, radius), tf.float32)
    f_radius = tf.cast(radius, tf.float32)
    #import matplotlib.pyplot as plt
    #plt.imshow(affine(padded_image.numpy(), theta=angle)[:, :, 0])
    #plt.show()
    for i, angle in enumerate(theta):
        #print(angle)
        # cos_a, sin_a = tf.math.cos(angle), tf.math.sin(angle)
        # R = [
            # [[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
            # [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
            # [0, 0, 1]]
        # ]
        # TODO: warp. we don't need full warp though, just this specific case
        # rotated = warp(padded_image, R, clip=False)
        # width, height = padded_image.shape[:2]
        flattened_image = tf.reshape(padded_image, [-1])
        #print(coords, coords.shape)
        #print('sparse', tf.map_fn(transformer(R), (flattened_image, coords))[1], flattened_image, padded_image.shape)
        #print('sparse', tf.sparse.SparseTensor(tf.map_fn(transformer(R), (flattened_image, coords))[1], flattened_image, padded_image.shape))
        #print(image, padded_image)
        #rotated = tf.map_fn(lambda r: padded_image[tf.cast(r * cos_a + f_radius, tf.int32), tf.cast(r * sin_a + f_radius, tf.int32)], diameter, tf.uint8)
        # technically using `affine` is numpy but it saves having to implement image manipulation in python which is slow
        result.append(tf.math.reduce_sum(affine(padded_image.numpy(), theta=angle), 0))
    result = tf.stack(result, 0)
    result = tf.math.abs(tf.signal.fft(tf.cast(result, tf.complex64)))
    result = tf.transpose(result, [1, 0, 2])
    import matplotlib.pyplot as plt
    result -= tf.math.reduce_min(result)
    result /= tf.math.reduce_max(result)
    print(result)
    plt.imshow(result[:, :, 0], cmap='gray')
    plt.show()
    return result

def transformer(R, w, h):
    #@tf.function
    def fn(args):
        value, coords = args
        x = coords[0]
        y = coords[1]
        multiplied = tf.linalg.matmul(R, tf.convert_to_tensor([[x], [y], [1]], tf.float32))
        x, y = multiplied[0, :2, 0]
        if x == 10: print(tf.cast([tf.clip_by_value(x, 0, w), tf.clip_by_value(y, 0, h)], tf.int32), int(y.numpy())) #float(multiplied[0][0][0].numpy()), float(multiplied[0][1][0].numpy()))
        return (value, tf.convert_to_tensor([tf.clip_by_value(x, 0, w), tf.clip_by_value(y, 0, h)], tf.int64))
    return fn
import matplotlib.pyplot as plt
def radon2(image, theta=None, circle=True, *, preserve_range=None):
    if len(image.shape) > 2:
        image = tf.image.rgb_to_grayscale(image)[:, :, 0]
    
    if theta is None:
        theta = tf.range(180)
    
    # fft2d = tf.signal.fft2d(tf.cast(image, tf.complex64))
    # r = tf.math.log(tf.math.real(fft2d))
    # g = tf.math.log(tf.math.imag(fft2d))
    # b = tf.zeros_like(r)
    # theta = tf.cast(theta, tf.float32) * math.pi / 180
    # r -= tf.reduce_min(r)
    # r /= tf.reduce_max(r)
    # g -= tf.reduce_min(g)
    # g /= tf.reduce_max(g)
    # print(tf.reduce_min(r), tf.reduce_max(r))
    # plt.imshow(tf.stack([r, g, b], 2))
    # plt.show()
    # fft2d = tf.signal.fft2d(tf.cast(image, tf.complex64))
    # r = tf.math.log(tf.signal.fftshift(tf.math.abs(fft2d)))
    # r -= tf.reduce_min(r)
    # r /= tf.reduce_max(r)
    # plt.imshow(r, cmap='gray')
    # plt.show()
    shape_min = tf.math.reduce_min(image.shape)
    radius = tf.math.floordiv(shape_min, 2)
    img_shape = image.shape
    x, y = tf.meshgrid(range(0, image.shape[0]), range(0, image.shape[1]))
    center_x, center_y = tf.math.floordiv(img_shape, 2)
    dist = (x - center_x) ** 2 + (y - center_y) ** 2
    outside_reconstruction_circle = dist > radius ** 2
    if tf.math.reduce_any(image[outside_reconstruction_circle] != 0):
        print('Radon transform: image must be zero outside the '
            'reconstruction circle')
    # Crop image to make it square
    slices = tuple(
        slice(int(tf.math.ceil(excess / 2).numpy()),
            int(tf.math.ceil(excess / 2) + shape_min).numpy())
        if excess > 0 else
        slice(None)
        for excess in (img_shape - shape_min)
    )
    padded_image = image[slices]
    center = tf.cast(tf.math.floordiv(padded_image.shape[0], 2), tf.float32)
    theta = tf.cast(theta, tf.float32) * math.pi / 180
    diameter = tf.cast(tf.range(-radius, radius), tf.float32)
    f_radius = tf.cast(radius, tf.float32)
    print(tf.math.reduce_mean(tf.cast(padded_image, tf.complex64)))
    fft2d = tf.signal.fft2d(tf.cast(padded_image, tf.complex64))
    shifted = tf.signal.fftshift(fft2d)
    radon_image = tf.map_fn(lambda r: tf.map_fn(lambda theta: shifted[tf.cast(r * tf.math.cos(theta) + f_radius, tf.int32), tf.cast(r * tf.math.sin(theta) + f_radius, tf.int32)], theta, tf.complex64), diameter, tf.complex64)
    ifft = tf.signal.ifft(radon_image)
    result = tf.math.abs(ifft)
    result -= tf.reduce_min(result)
    result /= tf.reduce_max(result)
    plt.imshow(result, cmap='gray')
    plt.show()
    v = tf.math.abs(tf.math.log(radon_image + 1))
    v -= tf.reduce_min(v)
    v /= tf.reduce_max(v)
    plt.imshow(v, cmap='gray')
    plt.show()

import tensorflow as tf; image = tf.image.decode_png(tf.io.read_file('test/phantom.png')); radon(image)