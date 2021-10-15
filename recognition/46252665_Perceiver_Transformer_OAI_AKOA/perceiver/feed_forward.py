import tensorflow as tf

from settings.config import *


class FeedForward(tf.keras.layers.Layer):
    """Tensorflow sequential dense layers with dropout in between."""

    @staticmethod
    def feed_forward_network():
        return tf.keras.Sequential([
                tf.keras.layers.Dense(units=FEED_FORWARD_NETWORK_UNITS[0],
                                      activation='relu'),
                tf.keras.layers.Dropout(DROPOUT),
                tf.keras.layers.Dense(units=FEED_FORWARD_NETWORK_UNITS[-1],
                                      use_bias=True, bias_initializer='zeros'),
            ]
        )
