"""
    gan.py

    This file contains the functions used to generate the generator and discriminator for StyleGAN.

    Requirements:
        - TensorFlow 2.0

    Author: Keith Dao
    Date created: 13/10/2021
    Date last modified: 13/10/2021
    Python version: 3.9.7
"""

import tensorflow as tf


def get_generator(
    input_dim: int, output_dim: tuple[int, int, int]
) -> tf.keras.Model:
    pass


def get_discriminator(
    input_dim: tuple[int, int, int], output_dim: int
) -> tf.keras.Model:
    pass
