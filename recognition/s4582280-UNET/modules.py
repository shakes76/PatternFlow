# Modules.py
# Contains code for UNET model components
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, UpSampling2D, Concatenate, Input, ZeroPadding2D
from tensorflow.keras.models import Model


"""
# A block which creates a double convolutional layer 
def ConvDouble(x, filters):
    x = layers.Conv2D(filters, 3, padding="same", activation="relu",
                      kernel_initializer="he_normal")(x)
    
    return (layers.Conv2D(filters, 3, padding="same", activation="relu",
                      kernel_initializer="he_normal")(x))
    
        
# Downsampling block for feature extraction
def Downsample(x, filters):
    block = ConvDouble(x, filters)
    pooling = layers.MaxPool2D(2)(block)
    pooling = layers.Dropout(0.35)(pooling)
    return block, pooling

# Upsampling block for decoding/expanding
def Upsample(x, features, filters):
    # Upsample
    x = layers.Conv2DTranspose(filters, 3, 2, padding="same")(x)
    # Concatenation
    x = layers.concatenate([x, features])
    # dropout layer
    x = layers.Dropout(0.35)(x)
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

    # Bottleneck layer
    p5 = ConvDouble(p4, 1024)

    # Decoding
    p6 = Upsample(p5, f4, 512)
    p7 = Upsample(p6, f3, 256)
    p8 = Upsample(p7, f2, 128)
    p9 = Upsample(p8, f1, 64)

    # Outputs
    outputs = layers.Conv2D(3, 1, padding="same", activation="relu")(p9)
    return tf.keras.Model(inputs, outputs, name="U-NET")
"""
    
    
