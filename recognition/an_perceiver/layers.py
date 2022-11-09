"""Perceiver layers.

@author Anthony North
"""

import tensorflow as tf
from tensorflow.keras import layers


class FeedForwardNetwork(layers.Layer):
    """Feed-forward network with gelu activation."""

    def __init__(self, name=None):
        super().__init__(name=name)

    def build(self, input_shape: tuple[int, ...]):
        units = input_shape[-1]
        self.dense1 = layers.Dense(
            units, activation=tf.nn.gelu, kernel_initializer="variance_scaling"
        )
        self.dense2 = layers.Dense(units, kernel_initializer="variance_scaling")

    def call(self, inputs: tf.Tensor):
        x = self.dense1(inputs)
        return self.dense2(x)


def layer_norm(epsilon=1e-5, **kwargs):
    """LayerNormalization layer with epsilon 1e-5."""
    return layers.LayerNormalization(epsilon=epsilon, **kwargs)


class CrossAttention(layers.Layer):
    """Cross-Attention followed by feed-forward network."""

    def __init__(self, num_heads: int = 1, key_dim: int = 1024, name=None):
        """
        Args:
            num_heads: Number of heads for the MultiHeadAttention layer.
            key_dim: Dimension of query and key attention heads.
            name: Layer name.
        """
        super().__init__(name=name)
        self.q_norm = layer_norm()
        self.kv_norm = layer_norm()
        self.attend = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.attn_norm = layer_norm()
        self.ffn = FeedForwardNetwork()

    def call(self, inputs_q: tf.Tensor, inputs_kv: tf.Tensor):
        q = self.q_norm(inputs_q)
        kv = self.kv_norm(inputs_kv)
        attend_result = self.attend(q, kv)
        attend_result += q

        z = self.ffn(self.attn_norm(attend_result))
        return z + attend_result


class SelfAttention(layers.Layer):
    """Self-Attention followed by feed-forward network."""

    def __init__(self, num_heads: int = 8, key_dim: int = 1024, name=None):
        """
        Args:
            num_heads: Number of heads for the MultiHeadAttention layer.
            key_dim: Dimension of query and key attention heads.
            name: Layer name.
        """
        super().__init__(name=name)
        self.qkv_norm = layer_norm()
        self.attend = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.attn_norm = layer_norm()
        self.ffn = FeedForwardNetwork()

    def call(self, inputs: tf.Tensor):
        qkv = self.qkv_norm(inputs)
        attend_result = self.attend(qkv, qkv)
        attend_result += inputs

        z = self.ffn(self.attn_norm(attend_result))
        return z + attend_result
