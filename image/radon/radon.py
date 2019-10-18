import tensorflow as tf
import math
from time import time

def radon(image, theta=None, circle=True, *, preserve_range=None):
    """
    Calculates the radon transform of an image given specified
    projection angles.
    """
    if len(image.shape) > 2:
        image = tf.image.rgb_to_grayscale(image)[:, :, 0]
    
    if theta is None:
        theta = tf.range(180)
    
    if circle:
        shape_min = tf.math.reduce_min(image.shape)
        radius = tf.math.floordiv(shape_min, 2)
        img_shape = image.shape
        x, y = tf.meshgrid(range(0, image.shape[0]), range(0, image.shape[1]))
        center_x, center_y = tf.math.floordiv(img_shape, 2)
        dist = (x - center_x) ** 2 + (y - center_y) ** 2
        outside_reconstruction_circle = dist > radius ** 2
        if tf.math.reduce_any(image[outside_reconstruction_circle] != 0):
            warn('Radon transform: image must be zero outside the '
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
    radon_image = tf.zeros((padded_image.shape[0], len(theta)))
    
    x, y = tf.meshgrid(tf.range(padded_image.shape[0], dtype=tf.int64), tf.range(padded_image.shape[1], dtype=tf.int64))
    x = tf.reshape(x, [-1])
    y = tf.reshape(y, [-1])
    coords = tf.stack([x, y], 1)
    result = []
    for i, angle in enumerate(tf.cast(theta, tf.float32) * math.pi / 180):
        cos_a, sin_a = tf.math.cos(angle), tf.math.sin(angle)
        R = tf.convert_to_tensor([
            [[cos_a, sin_a, -center * (cos_a + sin_a - 1)],
            [-sin_a, cos_a, -center * (cos_a - sin_a - 1)],
            [0, 0, 1]]
        ])
        # TODO: warp. we don't need full warp though, just this specific case
        # rotated = warp(padded_image, R, clip=False)
        width, height = padded_image.shape
        flattened_image = tf.reshape(padded_image, [-1])
        #print(coords, coords.shape)
        #print('sparse', tf.map_fn(transformer(R), (flattened_image, coords))[1], flattened_image, padded_image.shape)
        #print('sparse', tf.sparse.SparseTensor(tf.map_fn(transformer(R), (flattened_image, coords))[1], flattened_image, padded_image.shape))
        rotated = tf.sparse.to_dense(
            tf.sparse.SparseTensor(tf.map_fn(transformer(R), (flattened_image, coords))[1], flattened_image, padded_image.shape),
            validate_indices=False
        )
        # for x in range(width):
            # for y in range(height):
                # multiplied = tf.linalg.matmul(R, tf.convert_to_tensor([[x], [y], [1]], tf.float64))
                # x2, y2 = int(multiplied[0][0][0].numpy()), int(multiplied[0][1][0].numpy())
                # print(x2, y2, x, y)
                # rotated[x2, y2] += padded_image[x, y]
        result.append(tf.math.reduce_sum(rotated, 0))
    return tf.stack(result, 1)

def transformer(R):
    #@tf.function
    def fn(args):
        value, coords = args
        x = coords[0]
        y = coords[1]
        multiplied = tf.linalg.matmul(R, tf.convert_to_tensor([[x], [y], [1]], tf.float32))
        print(multiplied[0, :2, 0], end=' ')
        # if x == 0: print(tf.cast(multiplied[0, :2, 0], tf.int32), int(y.numpy())) #float(multiplied[0][0][0].numpy()), float(multiplied[0][1][0].numpy()))
        return (value, tf.cast(multiplied[0, :2, 0], tf.int64))
    return fn

import tensorflow as tf; image = tf.image.decode_png(tf.io.read_file('test/phantom.png')); radon(tf.image.resize(image, (40, 40)))