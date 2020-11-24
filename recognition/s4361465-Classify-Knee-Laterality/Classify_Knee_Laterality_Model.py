# Binary Classification model for COMP3710 recognition project
# Kelsey McGahan, s4361465

import tensorflow as tf


def classification(images_shape):
    """Model for binary classification 
    Parameters: 
          images_shape (tuple): The input shape of the images used to train the model."""

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape=images_shape),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    return model
