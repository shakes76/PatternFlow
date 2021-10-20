import tensorflow as tf
from tensorflow.keras import layers, initializers


def layer_norm(input):
    return layers.LayerNormalization(epsilon=1e-5)(input)


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
    """Cross-Attention followed by feed-forward network."""

    def __init__(self, num_heads: int = 1, name="cross_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads

    def build(self, input_q_shape: tuple[int, ...]):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=input_q_shape[-1]
        )
        self.feed_forward = FeedForwardNetwork()

    def call(self, inputs_q: tf.Tensor, inputs_kv: tf.Tensor):
        q = layer_norm(inputs_q)
        kv = layer_norm(inputs_kv)
        attend_result = self.attention(q, kv)
        return self.feed_forward(attend_result)


class SelfAttention(layers.Layer):
    """Self-Attention followed by feed-forward network."""

    def __init__(self, num_heads: int = 8, name="self_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads

    def build(self, input_shape: tuple[int, ...]):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=input_shape[-1]
        )
        self.feed_forward = FeedForwardNetwork()

    def call(self, inputs: tf.Tensor):
        qkv = layer_norm(inputs)
        attend_result = self.attention(qkv)
        return self.feed_forward(attend_result)


class ClassificationDecoder(layers.Layer):
    """Classification decoder block."""

    def __init__(self, num_classes: int = 2, name="logits"):
        super().__init__(name=name)
        self.logits = layers.Dense(num_classes, activation=tf.nn.softmax)

    def call(self, inputs: tf.Tensor):
        x = tf.reduce_mean(inputs, -2)
        return self.logits(x)
