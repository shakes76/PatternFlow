# !/user/bin/env python
"""
The generator and discriminator models of the StyleGAN
"""

import os
import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"


class Generator:
    def __init__(self, lr: float, beta_1: float, latent_dim: int, rgb=False):
        # hyper-parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.latent_dim = latent_dim
        self.rgb = rgb
        if self.rgb:
            self.channels = 3
        else:
            self.channels = 1

        # model
        self.model = None
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=self.beta_1)

    class ConvLayer(layers.Layer):
        def __init__(self, filters, kernel, stride):
            super(Generator.ConvLayer, self).__init__()
            self.conv1 = layers.Conv2DTranspose(filters, (kernel, kernel), strides=(stride, stride), padding='same',
                                                use_bias=False)
            self.bn = layers.BatchNormalization()
            self.activation = layers.LeakyReLU()

        def call(self, X):
            Y = self.activation(self.bn(self.conv1(X)))
            return Y

    def build(self):
        self.model = Sequential([
            # [100, ] to [8*8*256, ]
            layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(self.latent_dim,)),
            layers.BatchNormalization(),
            layers.LeakyReLU(),

            # [8*8*256, ] to [8, 8, 256]
            layers.Reshape((8, 8, 256)),

            # [8, 8, 256] to [16, 16, 128]
            self.ConvLayer(128, 5, 1),
            # to [32, 32, 64]
            self.ConvLayer(64, 5, 2),
            # to [64, 64, ]
            layers.Conv2DTranspose(self.channels, (5, 5), strides=(4, 4), padding='same', use_bias=False,
                                   activation='tanh')
        ])

    def loss(self, fake_score):
        return self._cross_entropy(tf.ones_like(fake_score), fake_score)


class Discriminator:
    def __init__(self, lr, beta_1, rgb=False):
        # hyper-parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.rgb = rgb
        if self.rgb:
            self.channels = 3
        else:
            self.channels = 1

        # model
        self.model = None
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam(self.lr, beta_1=self.beta_1)

    class ConvLayer(layers.Layer):
        def __init__(self, filters, kernel, stride):
            super(Discriminator.ConvLayer, self).__init__()
            self.conv = layers.Conv2D(filters, (kernel, kernel), strides=(stride, stride), padding='same')
            self.activation = layers.LeakyReLU()
            self.dropout = layers.Dropout(0.3)

        def call(self, X):
            Y = self.dropout(self.activation(self.conv(X)))
            return Y

    def build(self):
        self.model = Sequential([
            # [64, 64, ] to [16, 16, 64]
            layers.Conv2D(64, (5, 5), strides=(4, 4), padding='same', input_shape=[64, 64, self.channels]),
            layers.LeakyReLU(),
            layers.Dropout(0.3),

            # [8, 8, 128]
            self.ConvLayer(128, 5, 2),
            # [2]
            layers.Flatten(),
            layers.Dense(1)
        ])

    def loss(self, real_score, fake_score):
        real_loss = self._cross_entropy(tf.ones_like(real_score), real_score)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_score), fake_score)
        total_loss = real_loss + fake_loss
        return total_loss
