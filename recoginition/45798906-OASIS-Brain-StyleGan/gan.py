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
from tensorflow.keras.layers import (
    Activation,
    add,
    Conv2D,
    Conv2DTranspose,
    Dense,
    Input,
    Layer,
    LeakyReLU,
    Reshape,
)

# Custom layers
class AdaIN(Layer):
    """Adaptive Instance Normalisation Layer."""

    def __init__(self, epsilon: float = 1e-3):

        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape: tf.TensorShape) -> None:

        dim = input_shape[0][-1]
        if dim == None:
            raise ValueError(
                f"Excepted axis {-1} of the input tensor be defined, but got an input with shape {input_shape}."
            )

        super(AdaIn, self).build(input_shape)

    def call(self, inputs: tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """Apply the normalisation formula: gamma * (x - mean) / stddev + beta."""

        x, beta, gamma = inputs

        input_shape = x.shape
        axes = list(range(1, len(input_shape) - 1))
        mean = tf.math.reduce_mean(x, axes, keepdims=True)
        stddev = tf.math.reduce_std(x, axes, keepdims=True) + self.epsilon
        normalised = (x - mean) / stddev

        return normalised * gamma + beta


# Models
def get_generator(
    input_dim: int, output_dim: tuple[int, int, int]
) -> tf.keras.Model:
    pass


def get_discriminator(
    input_dim: tuple[int, int, int], output_dim: int
) -> tf.keras.Model:
    pass
