"""
Sequential Neural Network

@author: Pritish Roy
@email: pritish.roy@uq.edu.au
"""

import tensorflow as tf

from settings.config import *


class FeedForward(tf.keras.layers.Layer):
    """Tensorflow sequential dense layers with dropout in between."""

    @staticmethod
    def feed_forward_network():
        return tf.keras.Sequential([
                tf.keras.layers.Dense(units=PROJECTION_DIMENSION,
                                      activation='relu'),
                tf.keras.layers.Dropout(DROPOUT),
                tf.keras.layers.Dense(units=PROJECTION_DIMENSION,
                                      use_bias=True, bias_initializer='zeros'),
            ]
        )
