import tensorflow as tf
from tensorflow.keras import models
import layers


class Perceiver(models.Model):
    def __init__(
        self,
        num_blocks: int = 8,
        num_self_attends_per_block: int = 6,
        num_cross_heads: int = 1,
        num_self_attend_heads: int = 8,
        latent_dim: int = 512,
        latent_channels: int = 1024,
        num_freq_bands: int = 64,
        num_classes: int = 2,
        name: str = "perceiver",
    ):
        super().__init__(name=name)
        self.num_blocks = num_blocks

        assert latent_channels % num_cross_heads == 0
        assert latent_channels % num_self_attend_heads == 0

        self.latent = layers.Latent(dim=latent_dim, num_channels=latent_channels)
        self.fourier_enc = layers.FourierPositionEmbedding(num_bands=num_freq_bands)

        cross_attention = lambda: layers.CrossAttention(
            num_heads=num_cross_heads, key_dim=latent_channels // num_cross_heads
        )
        self_attentions = lambda: [
            layers.SelfAttention(
                num_heads=num_self_attend_heads,
                key_dim=latent_channels // num_self_attend_heads,
            )
            for _ in range(num_self_attends_per_block)
        ]

        # first block doesn't share weights with following blocks
        self.initial_block = cross_attention(), self_attentions()
        self.repeat_block = cross_attention(), self_attentions()

        self.logits = layers.ClassificationDecoder(num_classes=num_classes)

    def call(self, inputs: tf.Tensor):
        latent_array = self.latent(inputs)
        inputs_enc = self.fourier_enc(inputs)

        # initial block
        cross_attention, self_attentions = self.initial_block
        z = cross_attention(latent_array, inputs_enc)
        for self_attention in self_attentions:
            z = self_attention(z)

        # repeats for 2 -> num_blocks
        cross_attention, self_attentions = self.repeat_block
        for _ in range(1, self.num_blocks):
            z = cross_attention(z, inputs_enc)
            for self_attention in self_attentions:
                z = self_attention(z)

        return self.logits(z)
