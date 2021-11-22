"""Fourier position encoding.

@author Anthony North
"""

import tensorflow as tf
import math


def fourier_position_encode(index_shape: tuple[int, ...], num_bands: int):
    """Uniformly sampled Fourier frequency position encodings.

    Returns a tensor of `(num_pixels, len(index_shape) * (2 * num_bands + 1))`
    where `num_pixels` = unrolled length of `index_shape`.

    Args:
        index_shape: Shape / dimensions of the indices.
        num_bands: Number of fourier frequency bands.
    """

    num_pixels = tf.reduce_prod(index_shape)
    positions = get_spatial_positions(index_shape)
    bands = tf.stack([tf.linspace(1.0, dim / 2, num_bands) for dim in index_shape])

    freq = tf.reshape(
        # each postion * each band
        positions[:, :, tf.newaxis] * bands,
        # flatten to pixels x [len(index_shape) * num_bands]
        [num_pixels, -1],
    )

    encodings = tf.concat(
        [
            tf.sin(math.pi * freq),
            tf.cos(math.pi * freq),
        ],
        axis=-1,
    )

    return tf.concat([positions, encodings], axis=-1)


def get_spatial_positions(index_shape: tuple[int, ...]):
    """Get spatial positions between -1 and 1.

    Returns a tensor of `(num_pixels, len(index_shape))`,
    where `num_pixels` = unrolled length of `index_shape`.

    Args:
        index_shape: Shape / dimensions of the indices.
    """

    coords = [tf.linspace(-1.0, 1.0, dim) for dim in index_shape]
    pos = tf.stack(tf.meshgrid(*coords, indexing="ij"), axis=-1)

    return tf.reshape(pos, [tf.reduce_prod(index_shape), -1])
