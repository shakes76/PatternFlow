# !/user/bin/env python
"""
The generator and discriminator models of the StyleGAN
"""

import os
import tensorflow as tf
from tensorflow.keras import Sequential

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"


def pixel_norm(x):
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True))


class Generator:
    def __init__(self):
        self.mapping = Sequential()
        self.synthesis = Sequential()

    def build_mapping(self, latent, labels, latent_size: int, label_size):
        """
        Construct the generator model according to the Figure 1 in the paper
        """
        dtype = "float32"

        # process the inputs
        latent.set_shape([None, latent_size])
        labels.set_shape([None, label_size])
        latents_in = tf.cast(latent, dtype)
        labels_in = tf.cast(labels, dtype)
        x = latents_in

        # Embed labels and concatenate them with latents if given
        if label_size:
            with tf.variable_scope('LabelConcat'):
                w = tf.get_variable('weight', shape=[label_size, latent_size],
                                    initializer=tf.initializers.random_normal())
                y = tf.matmul(labels_in, tf.cast(w, dtype))
                x = tf.concat([x, y], axis=1)

        x = pixel_norm(x)
