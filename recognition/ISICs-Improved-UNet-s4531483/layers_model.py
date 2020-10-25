import tensorflow as tf
from tensorflow import keras


def improvedUNet(width, height, channels, classes=2):
    input = keras.Input(shape=(width, height, channels))
    x1 = keras.layers.Conv2D(32, (3, 3), padding="same", input_shape=(width, height, channels))(input)
    x2 = keras.layers.Activation("relu")(x1)
    x3 = keras.layers.Flatten()(x2)
    x4 = keras.layers.Dense(classes)(x3)
    output = keras.layers.Activation("softmax")(x4)
    unet = keras.Model(inputs=[input], outputs=[output])
    return unet