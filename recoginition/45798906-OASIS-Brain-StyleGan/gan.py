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
    Dense,
    Input,
    Layer,
    LeakyReLU,
    Reshape,
    UpSampling2D,
)

# Custom layers
class AdaIN(Layer):
    """Adaptive Instance Normalisation Layer."""

    def __init__(self, epsilon: float = 1e-3):

        super(AdaIN, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape: list[tf.TensorShape]) -> None:

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


# Layer blocks
def gen_block(
    input: tf.Tensor, style: tf.Tensor, noise: tf.Tensor, filters: int
) -> tf.Tensor:
    """
    For each block, we want to: (In order)
        - Upscale
        - Conv 3x3
        - Add noise
        - AdaIN
        - Conv 3x3
        - Add noise
        - AdaIN
    """

    beta = Dense(filters)(style)
    beta = Reshape([1, 1, filters])(beta)
    gamma = Dense(filters)(style)
    gamma = Reshape([1, 1, filters])(gamma)
    n = Conv2D(filters, kernel_size=1, padding="same")(noise)

    # Begin the generator block
    out = UpSampling2D(interpolation="bilinear")(input)
    out = Conv2D(filters, kernel_size=3, padding="same")(out)
    out = add([out, n])
    out = AdaIN()([out, beta, gamma])
    out = LeakyReLU(0.01)(out)

    # Compute new beta, gamma and noise
    beta = Dense(filters)(style)
    beta = Reshape([1, 1, filters])(beta)
    gamma = Dense(filters)(style)
    gamma = Reshape([1, 1, filters])(gamma)
    n = Conv2D(filters, kernel_size=1, padding="same")(noise)

    # Continue the generator block
    out = Conv2D(filters, kernel_size=3, padding="same")(out)
    out = add([out, n])
    out = AdaIN()([out, beta, gamma])
    out = LeakyReLU(0.01)(out)

    return out


# Models
def get_generator(
    input_dim: int, output_dim: tuple[int, int, int]
) -> tf.keras.Model:
    pass


def get_discriminator(
    input_dim: tuple[int, int, int], output_dim: int
) -> tf.keras.Model:
    pass
