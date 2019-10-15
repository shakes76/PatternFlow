import tensorflow as tf


def downscale_local_mean(image, factors, cval=0, clip=True):
    return block_reduce(image, factors, tf.reduce_mean, cval)

