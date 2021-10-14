"""
    train.py

    This file contains functions that enables GAN training.

    Requirements:
        - Tensorflow 2.0
        - gan.py
    
    Author: Keith Dao
    Date created: 14/10/2021
    Date last modified: 14/10/2021
    Python version: 3.9.7
"""

import tensorflow as tf

# Loss functions
def generator_loss(fakes: tf.Tensor) -> float:

    return tf.keras.losses.BinaryCrossEntropy(tf.ones_like(fakes), fakes)


def discriminator_loss(reals: tf.Tensor, fakes: tf.Tensor) -> float:

    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return (
        cross_entropy(tf.ones_like(reals), reals)
        + cross_entropy(tf.zeros_like(fakes), fakes)
    ) / 2
