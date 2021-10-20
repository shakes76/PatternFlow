import tensorflow as tf
from tensorflow.keras import models
import layers


class Perceiver(models.Model):
    def __init__(
        self,
        num_blocks=8,
        num_self_attends_per_block=6,
        num_cross_heads=1,
        num_self_attend_heads=8,
        latent_dim=512,
        latent_channels=1024,
        num_freq_bands=64,
        num_classes=2,
        name="perceiver",
    ):
        super().__init__(name=name)
        self.latent = layers.Latent(dim=latent_dim, num_channels=latent_channels)
        self.cross_attention = layers.CrossAttention(num_heads=num_cross_heads)
        self.self_attend = layers.SelfAttention(num_heads=num_self_attend_heads)
        self.logits = layers.ClassificationDecoder(num_classes=num_classes)
