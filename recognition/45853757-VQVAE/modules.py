import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

import numpy as np

length = 256*256
batch_size = 32

depth = 16
kernel = 3


def create_encoder(latent_dim=16):
    """ Create a simple encoder """
    encoder = layers.Sequential(name="encoder")
    encoder.add(layers.Conv2D(depth, kernel, activation="relu", strides=2, padding="same", input_shape=(length,)))
    encoder.add(layers.Conv2D(depth*2, kernel, activation="relu", strides=2, padding="same"))
    encoder.add(layers.Conv2D(depth*4, kernel, activation="relu", strides=2, padding="same"))
    encoder.add(layers.Conv2D(depth*8, kernel, activation="relu", strides=2, padding="same"))
    encoder.add(layers.Conv2D(latent_dim, 1, padding="same"))
    return encoder
        
def create_decoder():
    """ Create a simple decoder """
    decoder = layers.Sequential(name="decoder")
    decoder.add(layers.Conv2D(depth*8, kernel, activation="relu", strides=2, padding="same"))
    decoder.add(layers.Conv2D(depth*4, kernel, activation="relu", strides=2, padding="same"))
    decoder.add(layers.Conv2D(depth*2, kernel, activation="relu", strides=2, padding="same"))
    decoder.add(layers.Conv2D(depth, kernel, activation="relu", strides=2, padding="same"))
    decoder.add(layers.Conv2D(1, kernel, padding="same"))
    return decoder


class VQLayer(layers.Layer):
    pass
    # Create the vector quantization layer for the model
    

class VQVAEModel(models.Sequential):
    pass
    # Use Sequential as base since i'm using sequential for encoder/decoder. 
    # Source I'm referencing uses models.Model so adapt appropriately

class PixelCNN:
    pass
    # Work out whether this needs to be implemented here or in a diff section/helper methods etc

def do_training():
    pass
