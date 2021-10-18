import tensorflow as tf
import math


def fourier_position_encode(input_shape: tuple[int, int, int], num_bands: int):
    """Uniformly sampled Fourier frequency position encodings."""

    # drop num_channels from input_shape
    index_shape = input_shape[:-1]

    positions = get_spatial_positions(index_shape)
    bands = tf.stack([tf.linspace(1.0, dim / 2, num_bands) for dim in index_shape])

    rads = math.pi * tf.reshape(
        # each postion * each band
        positions[:, :, tf.newaxis] * bands,
        [-1, tf.reduce_prod(bands.shape)],
    )

    encodings = tf.concat([tf.sin(rads), tf.cos(rads)], axis=-1)
    return tf.concat([positions, encodings], axis=-1)


def get_spatial_positions(index_shape: tuple[int, int]):
    """Get spatial positions between -1 and 1."""

    coords = [tf.linspace(-1.0, 1.0, dim) for dim in index_shape]
    pos = tf.stack(tf.meshgrid(*coords, indexing="ij"), axis=-1)

    return tf.reshape(pos, [tf.reduce_prod(index_shape), -1])
