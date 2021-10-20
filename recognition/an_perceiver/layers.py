import tensorflow as tf
from tensorflow.keras import layers, initializers


class Latent(layers.Layer):
    """Trainable latent array, initialised with truncated normal."""

    def __init__(self, dim: int, num_channels: int, name: str = "latent"):
        super().__init__(name=name)
        self.latent_array = self.add_weight(
            shape=(dim, num_channels),
            initializer=initializers.TruncatedNormal(stddev=0.02),
        )

    def call(self, inputs: tf.Tensor):
        batch_size = inputs.shape[0]

        return (
            self.latent_array
            if batch_size is None
            else tf.broadcast_to(
                self.latent_array, [batch_size, *self.latent_array.shape]
            )
        )


class FeedForwardNetwork(layers.Layer):
    """Feed-forward network with gelu activation."""

    def __init__(self, name="feed_forward"):
        super().__init__(name=name)

    def build(self, input_shape: tuple[int, ...]):
        dense = lambda dim: layers.Dense(
            dim, kernel_initializer=initializers.VarianceScaling()
        )

        dim = input_shape[-1]
        self.dense1 = dense(dim)
        self.dense2 = dense(dim)

    def call(self, inputs: tf.Tensor):
        x = self.dense1(inputs)
        x = tf.nn.gelu(x)
        return self.dense2(x)


class CrossAttention(layers.Layer):
    def __init__(self, num_heads=1, name="cross_attention"):
        super().__init__(name=name)


class SelfAttention(layers.Layer):
    def __init__(self, num_heads=8, name="self_attention"):
        super().__init__(name=name)


class ClassificationDecoder(layers.Layer):
    def __init__(self, num_classes=2, name="logits"):
        super().__init__(name=name)
