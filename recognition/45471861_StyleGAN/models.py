# !/user/bin/env python
"""
The generator and discriminator models of the StyleGAN
"""

import os
import tensorflow as tf
from tensorflow.keras import Sequential, layers

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"


class Generator:
    def __init__(self, alpha: float, lr: float, beta_1: float, latent_dim: int, rgb=False):
        # hyper-parameters
        self.alpha = alpha
        self.lr = lr
        self.beta_1 = beta_1
        self.latent_dim = latent_dim
        self.rgb = rgb

        # model
        self.model = Sequential()
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=self.beta_1)

    def build(self):
        # [100, ] to [8*8*256, ]
        self.model.add(layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(self.latent_dim,)))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=self.alpha))

        # [8*8*256, ] to [8, 8, 256]
        self.model.add(layers.Reshape((8, 8, 256)))

        # [8, 8, 256] to [16, 16, 128]
        self.model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=self.alpha))

        # to [32, 32, 64]
        self.model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
        self.model.add(layers.BatchNormalization())
        self.model.add(layers.LeakyReLU(alpha=self.alpha))

        if self.rgb:
            channels = 3
        else:
            channels = 1

        # to [64, 64, ]
        self.model.add(
            layers.Conv2DTranspose(channels, (5, 5), strides=(4, 4), padding='same', use_bias=False, activation='tanh'))

    def loss(self, fake_score):
        return self._cross_entropy(tf.ones_like(fake_score), fake_score)


class Discriminator:
    def __init__(self, dropout, lr, beta_1, rgb=False):
        # hyper-parameters
        self.dropout = dropout
        self.lr = lr
        self.beta_1 = beta_1
        self.rgb = rgb

        # model
        self.model = Sequential()
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=self.beta_1)

    def build(self):
        if self.rgb:
            channels = 3
        else:
            channels = 1

        # [64, 64, ] to [16, 16, 64]
        self.model.add(layers.Conv2D(64, (5, 5), strides=(4, 4), padding='same', input_shape=[64, 64, channels]))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(self.dropout))

        # [8, 8, 128]
        self.model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
        self.model.add(layers.LeakyReLU())
        self.model.add(layers.Dropout(self.dropout))

        # [2]
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(1))

    def loss(self, real_score, fake_score):
        real_loss = self._cross_entropy(tf.ones_like(real_score), real_score)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_score), fake_score)
        total_loss = real_loss + fake_loss
        return total_loss
