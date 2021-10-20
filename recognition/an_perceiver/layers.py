from tensorflow.keras import layers


class Latent(layers.Layer):
    def __init__(self, dim: int, num_channels: int, name="latent"):
        super().__init__(name=name)


class FeedForwardNetwork(layers.Layer):
    def __init__(self, name="feed_forward"):
        super().__init__(name=name)


class CrossAttention(layers.Layer):
    def __init__(self, num_heads=1, name="cross_attention"):
        super().__init__(name=name)

        layers.Dense


class SelfAttention(layers.Layer):
    def __init__(self, num_heads=8, name="self_attention"):
        super().__init__(name=name)


class ClassificationDecoder(layers.Layer):
    def __init__(self, num_classes=2, name="logits"):
        super().__init__(name=name)
