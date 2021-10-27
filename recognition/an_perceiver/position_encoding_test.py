import tensorflow as tf
import position_encoding
import math


def test_spatial_pos():
    index_dims = (2, 3)
    positions = position_encoding.get_spatial_positions(index_dims)

    # each pixel contains position for each index
    num_pixels = tf.reduce_prod(index_dims)
    num_indices = len(index_dims)
    assert positions.shape == (num_pixels, num_indices)
    assert tf.reduce_min(positions) == -1
    assert tf.reduce_max(positions) == 1

    expected_pos = tf.constant(
        [
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )

    assert tf.reduce_all(positions == expected_pos)


def test_fourier_pos():
    index_dims = (2, 3)
    num_bands = 2
    num_indices = len(index_dims)
    num_pos = num_indices * num_bands
    fourier_pos = position_encoding.fourier_position_encode(index_dims, num_bands)

    # each pixel contains raw spatial position + fourier encodings
    # expected channels = len(num_indices) * (2 * num_bands + 1)
    fourier_channels = len(index_dims) * (2 * num_bands + 1)
    num_pixels = tf.reduce_prod(index_dims)
    assert fourier_pos.shape == (num_pixels, fourier_channels)

    # each pixel contains
    # - raw pos, sin_encodings, cos_encodings
    raw_pos, sin_enc, cos_enc = tf.split(
        fourier_pos, [num_indices, num_pos, num_pos], axis=-1
    )

    expected_raw_pos = tf.constant(
        [
            [-1.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    assert tf.reduce_all(raw_pos == expected_raw_pos)

    # multiply each position by the fouier bands
    freq = tf.constant(
        [
            [-1.0, -1.0, -1.0, -1.5],
            [-1.0, -1.0, 0.0, 0.0],
            [-1.0, -1.0, 1.0, 1.5],
            [1.0, 1.0, -1.0, -1.5],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.5],
        ]
    )

    expected_sin_enc = tf.sin(math.pi * freq)
    expected_cos_enc = tf.cos(math.pi * freq)

    assert tf.reduce_all(sin_enc == expected_sin_enc)
    assert tf.reduce_all(cos_enc == expected_cos_enc)


test_spatial_pos()
test_fourier_pos()
