# !/user/bin/env python
"""
The generator and discriminator models of the StyleGAN
"""

import os
import tensorflow as tf
from tensorflow.keras import Sequential, layers, Model

__author__ = "Zhien Zhang"
__email__ = "zhien.zhang@uqconnect.edu.au"


class _Generator(Model):
    class ConvLayer(layers.Layer):
        def __init__(self, filters, kernel, stride):
            super(_Generator.ConvLayer, self).__init__()
            self.conv1 = layers.Conv2DTranspose(filters, (kernel, kernel), strides=(stride, stride), padding='same',
                                                use_bias=False)
            self.bn = layers.BatchNormalization()
            self.activation = layers.LeakyReLU()

        def call(self, X):
            Y = self.activation(self.bn(self.conv1(X)))
            return Y

    def __init__(self, latent_dim, channels):
        super().__init__()
        # [100, ] to [8*8*256, ]
        self.dense = layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(latent_dim,))
        self.bn1 = layers.BatchNormalization()
        self.ac1 = layers.LeakyReLU()

        # [8*8*256, ] to [8, 8, 256]
        self.reshape = layers.Reshape((8, 8, 256))

        # [8, 8, 256] to [16, 16, 128]
        self.conv1 = self.ConvLayer(128, 5, 1)
        # to [32, 32, 64]
        self.conv2 = self.ConvLayer(64, 5, 2)
        # to [64, 64, ]
        self.conv3 = layers.Conv2DTranspose(channels, (5, 5), strides=(4, 4), padding='same', use_bias=False,
                                            activation='tanh')

    def call(self, X):
        Y = self.ac1(self.bn1(self.dense(X)))
        Y = self.reshape(Y)
        Y = self.conv1(Y)
        Y = self.conv2(Y)
        Y = self.conv3(Y)
        return Y


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

    class ToRGB(Model):
        def __init__(self, num_channels):
            super().__init__()
            self.conv = tf.keras.layers.Conv2D(num_channels, (1, 1), strides=(1, 1), padding='same')

        def call(self, X):
            return self.conv(X)

    def build(self):
        self.model = _Generator(self.latent_dim, self.channels)

    def loss(self, fake_score):
        return self._cross_entropy(tf.ones_like(fake_score), fake_score)


class _Discriminator(Model):
    class ConvLayer(layers.Layer):
        def __init__(self, filters, kernel, stride):
            super(_Discriminator.ConvLayer, self).__init__()
            self.conv = layers.Conv2D(filters, (kernel, kernel), strides=(stride, stride), padding='same')
            self.activation = layers.LeakyReLU()
            self.dropout = layers.Dropout(0.3)

        def call(self, X):
            Y = self.dropout(self.activation(self.conv(X)))
            return Y

    def __init__(self, channels):
        super().__init__()
        # [64, 64, ] to [16, 16, 64]
        self.conv1 = layers.Conv2D(64, (5, 5), strides=(4, 4), padding='same', input_shape=[64, 64, channels])
        self.ac1 = layers.LeakyReLU()
        self.dropout1 = layers.Dropout(0.3)

        # [8, 8, 128]
        self.conv2 = self.ConvLayer(128, 5, 2)
        # [1]
        self.flatten = layers.Flatten()
        self.dense = layers.Dense(1)

    def call(self, X):
        Y = self.dropout1(self.ac1(self.conv1(X)))
        Y = self.conv2(Y)
        Y = self.flatten(Y)
        Y = self.dense(Y)
        return Y


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

    def build(self):
        self.model = _Discriminator(self.channels)

    def loss(self, real_score, fake_score):
        real_loss = self._cross_entropy(tf.ones_like(real_score), real_score)
        fake_loss = self._cross_entropy(tf.zeros_like(fake_score), fake_score)
        total_loss = real_loss + fake_loss
        return total_loss
