"""
Architecture for Improved Unet

@author Lachlan Taylor
"""
from tensorflow import keras

"""
Initial layers for improved unet model
"""
def improved_unet(height, width):
    inputs = keras.layers.Input((height, width, 1))
    outputs = keras.layers.Conv2D(1, 1, activation = 'sigmoid')(inputs)
    model = keras.Model(inputs = inputs, outputs = outputs)
    return model