import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

def get_model(upscale_factor=4, channels=1):
    inputs = keras.Input(shape=(None, None, 1))
    x = layers.Conv2D(64, 5, activation = "leaky_relu", kernel_initializer = "Orthogonal", padding = "same")(inputs)
    x = layers.Conv2D(64, 3, activation = "leaky_relu", kernel_initializer = "Orthogonal", padding = "same")(x)
    x = layers.Conv2D(32, 3, activation = "leaky_relu", kernel_initializer = "Orthogonal", padding = "same")(x)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, activation = "leaky_relu", kernel_initializer = "Orthogonal", padding = "same")(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)
