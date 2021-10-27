"""Perceiver classifier model.

@author Anthony North
"""

import tensorflow as tf
from tensorflow.keras import models, layers, initializers
from layers import CrossAttention, SelfAttention
from position_encoding import fourier_position_encode


class Perceiver(models.Model):
    """Perceiver: General Perception with Iterative Attention."""

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
        """Initialises a Perceiver model.

        All parameters have defaults that are consistent with the configuration
        described in the Perceiver paper https://arxiv.org/abs/2103.03206 for
        imagenet classification.

        Args:
            num_blocks: Number of cross attention and self attention blocks.
                Each block consists of a cross attention layer and
                `num_self_attends_per_block` self attention layers.
            num_self_attends_per_block: Number of self attention layers per block.
            num_cross_heads: Number of heads for the cross attention layers.
            num_self_attend_heads: Number of heads for the self attention layers.
            latent_dim: Dimension (axis 0) of the latent array.
            latent_channels: Channels (axis 1) of the latent array.
            num_freq_bands: Number of fourier frequency bands.
                Fourier position encodings are concatenated with the input data.
            num_classes: Number of classes for the classifier.
                This determines the output shape (batch_size, num_classes)
            name: Name of the model.
        """
        super().__init__(name=name)
        assert latent_channels % num_cross_heads == 0
        assert latent_channels % num_self_attend_heads == 0

        self.num_blocks = num_blocks
        self.num_self_attends_per_block = num_self_attends_per_block
        self.num_cross_heads = num_cross_heads
        self.num_self_attend_heads = num_self_attend_heads
        self.latent_dim = latent_dim
        self.latent_channels = latent_channels
        self.num_freq_bands = num_freq_bands
        self.num_classes = num_classes

    def build(self, input_shape: tuple[int, ...]):
        self.latent_array = self.add_weight(
            name="latent_array",
            shape=(self.latent_dim, self.latent_channels),
            initializer=initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

        self.cross_attend = CrossAttention(
            num_heads=self.num_cross_heads, key_dim=self.latent_channels
        )

        self.self_attends = [
            SelfAttention(
                num_heads=self.num_self_attend_heads, key_dim=self.latent_channels
            )
            for _ in range(self.num_self_attends_per_block)
        ]

        self.logits = layers.Dense(self.num_classes)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor):
        batch_size = tf.shape(inputs)[0]
        index_shape = inputs.shape[1:-1]
        channels = inputs.shape[-1]

        # broadcast to batch
        broadcast = lambda x: tf.broadcast_to(x, [batch_size, *x.shape])
        latent_array = broadcast(self.latent_array)
        fourier_pos = broadcast(
            fourier_position_encode(index_shape, self.num_freq_bands)
        )

        # flatten index dims, concat with fourier position
        inputs_vec = tf.reshape(
            inputs, [batch_size, tf.reduce_prod(index_shape), channels]
        )
        data_array = tf.concat([inputs_vec, fourier_pos], axis=-1)

        # apply blocks
        for _ in range(self.num_blocks):
            latent_array = self.cross_attend(latent_array, data_array)
            for self_attend in self.self_attends:
                latent_array = self_attend(latent_array)

        x = tf.reduce_mean(latent_array, axis=-2)
        return self.logits(x)
