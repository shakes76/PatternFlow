import keras
import Tensorflow as tf
from keras import layers

'''
Three convolutional layers and two activation layers,
with the final layer being a sub-pixel shuffle function
'''
def model(upscale_factor=4):
    inputs = keras.Input(shape=(None, None, 1))
    x = layers.Conv2D(64, 5, padding="same", activation="tanh",
		kernel_initializer="Orthogonal" )(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="tanh",
		kernel_initializer="Orthogonal")(x)
    x = layers.Conv2D(1 * (upscale_factor ** 2), 3, padding="same",
		kernel_initializer="Orthogonal")(x)

    outputs = tf.nn.depth_to_space(x, upscale_factor)

    model = keras.Model(inputs, outputs)
    return model