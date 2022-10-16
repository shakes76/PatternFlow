# Modules.py
# Contains code for UNET model components
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

# A block which creates a double convolutional layer 
def ConvDouble(x, filters):
    x = layers.Conv2D(filters, 3, activation="relu",
                      kernel_initializer="he_normal")(x)
    return layers.Conv2D(filters, 3, activation="relu",
                      kernel_initializer="he_normal")(x)
        
# Downsampling block for feature extraction
def Downsample(x, filters):
    block = ConvDouble(x, filters)
    pooling = layers.MaxPool2D(2)(block)
    pooling = layers.Dropout(0.25)(pooling)
    return block, pooling

# Upsampling block for decoding/expanding
def Upsample(x, features, filters):
    # Upsample
    x = layers.Conv2DTranspose(filters, 3, 3, padding="same")(x)
    # Concatenation
    x = layers.concatenate([x, features])
    # dropout layer
    x = layers.Dropout(0.25)(x)
    # Double convolutional layers
    return ConvDouble(x, filters)

# Function that creates and returns the UNET model
def BuildUNET():
    # Define image dimensions to be added
    inputs = layers.Input(shape=(128, 128, 3))

    # Encoding and downsampling
    f1, p1 = Downsample(inputs, 64)
    f2, p2 = Downsample(p1, 128)
    f3, p3 = Downsample(p2, 256)
    f4, p4 = Downsample(p3, 512)
    
