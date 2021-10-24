import tensorflow as tf
from tensorflow.keras import layers, initializers
from position_encoding import fourier_position_encode


class Latent(layers.Layer):
    """Trainable latent array, initialised with truncated normal."""

    def __init__(self, dim: int, num_channels: int, name: str = "latent"):
        super().__init__(name=name)
        self.latent_array = self.add_weight(
            name="latent",
            shape=(dim, num_channels),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, inputs: tf.Tensor):
        batch_size = tf.shape(inputs)[0]

        # broadcast to batch
        return tf.broadcast_to(
            self.latent_array, [batch_size, *self.latent_array.shape]
        )


class FourierPositionEmbedding(layers.Layer):
    """Fourier position encoding."""

    def __init__(self, num_bands: int = 64, name: str = "fourier_encode"):
        super().__init__(name=name)
        self.num_bands = num_bands

    def build(self, input_shape: tuple[int, ...]):
        index_shape = input_shape[1:-1]
        self.pos_encodings = fourier_position_encode(index_shape, self.num_bands)

    def call(self, inputs: tf.Tensor):
        batch_size = tf.shape(inputs)[0]
        index_shape = inputs.shape[1:-1]
        dim = inputs.shape[-1]

        # broadcast to batch
        pos_encodings = tf.broadcast_to(
            self.pos_encodings, [batch_size, *self.pos_encodings.shape]
        )

        # unroll image
        inputs_unrolled = tf.reshape(
            inputs, [batch_size, tf.reduce_prod(index_shape), dim]
        )

        return tf.concat([inputs_unrolled, pos_encodings], axis=-1)


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


def layer_norm(epsilon=1e-5, **kwargs):
    return layers.LayerNormalization(epsilon=epsilon, **kwargs)


class CrossAttention(layers.Layer):
    """Cross-Attention followed by feed-forward network."""

    def __init__(self, num_heads: int = 1, name="cross_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads

        self.q_norm = layer_norm()
        self.kv_norm = layer_norm()
        self.residual_add = layers.Add()
        self.add_norm = layer_norm()
        self.feed_forward = FeedForwardNetwork()

    def build(self, input_q_shape: tuple[int, ...]):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=input_q_shape[-1]
        )

    def call(self, inputs_q: tf.Tensor, inputs_kv: tf.Tensor):
        q = self.q_norm(inputs_q)
        kv = self.kv_norm(inputs_kv)
        attend_result = self.attention(q, kv)
        x = self.residual_add([inputs_q, attend_result])

        return self.feed_forward(self.add_norm(x))


class SelfAttention(layers.Layer):
    """Self-Attention followed by feed-forward network."""

    def __init__(self, num_heads: int = 8, name="self_attention"):
        super().__init__(name=name)
        self.num_heads = num_heads

        self.qkv_norm = layer_norm()
        self.residual_add = layers.Add()
        self.add_norm = layer_norm()
        self.feed_forward = FeedForwardNetwork()

    def build(self, input_shape: tuple[int, ...]):
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=input_shape[-1]
        )

    def call(self, inputs: tf.Tensor):
        qkv = self.qkv_norm(inputs)
        attend_result = self.attention(qkv, qkv)
        x = self.residual_add([inputs, attend_result])

        return self.feed_forward(self.add_norm(x))


class ClassificationDecoder(layers.Layer):
    """Classification decoder block."""

    def __init__(self, num_classes: int = 2, name="logits"):
        super().__init__(name=name)
        self.dense = layers.Dense(num_classes)

    def call(self, inputs: tf.Tensor):
        x = tf.reduce_mean(inputs, axis=-2)
        return self.dense(x)
