"""
Creates and returns a standard UNET model using tensorflow.

@author Max Hornigold
"""

import tensorflow as tf
from tensorflow.keras import layers

def unet_model(f=16):
    """Creates and returns a standard UNET model. User is able to specify
    the number of filters using f, although f=16 is recommended."""
    
    # create an input layer
    inputs = tf.keras.layers.Input(shape=(256, 256, 1))
    
    # downsample (encoder)
    d1 = layers.Conv2D(f, 3, padding='same', activation='relu')(inputs)
    d1 = layers.Conv2D(f, 3, padding='same', activation='relu')(d1)
    
    d2 = layers.MaxPooling2D()(d1)
    d2 = layers.Conv2D(2*f, 3, padding='same', activation='relu')(d2)
    d2 = layers.Conv2D(2*f, 3, padding='same', activation='relu')(d2)
    
    d3 = layers.MaxPooling2D()(d2)
    d3 = layers.Conv2D(4*f, 3, padding='same', activation='relu')(d3)
    d3 = layers.Conv2D(4*f, 3, padding='same', activation='relu')(d3)
      
    d4 = layers.MaxPooling2D()(d3)
    d4 = layers.Conv2D(8*f, 3, padding='same', activation='relu')(d4)
    d4 = layers.Conv2D(8*f, 3, padding='same', activation='relu')(d4)
                               
    d5 = layers.MaxPooling2D()(d4)
    d5 = layers.Conv2D(16*f, 3, padding='same', activation='relu')(d5)
    d5 = layers.Conv2D(16*f, 3, padding='same', activation='relu')(d5)
    
    # upsample (decoder)
    u4 = layers.UpSampling2D()(d5)
    u4 = layers.concatenate([u4, d4])
    u4 = layers.Conv2D(8*f, 3, padding='same', activation='relu')(u4)
    u4 = layers.Conv2D(8*f, 3, padding='same', activation='relu')(u4)
    
    u3 = layers.UpSampling2D()(u4)
    u3 = layers.concatenate([u3, d3])
    u3 = layers.Conv2D(4*f, 3, padding='same', activation='relu')(u3)
    u3 = layers.Conv2D(4*f, 3, padding='same', activation='relu')(u3)
    
    u2 = layers.UpSampling2D()(u3)
    u2 = layers.concatenate([u2, d2])
    u2 = layers.Conv2D(2*f, 3, padding='same', activation='relu')(u2)
    u2 = layers.Conv2D(2*f, 3, padding='same', activation='relu')(u2)
    
    u1 = layers.UpSampling2D()(u2)
    u1 = layers.concatenate([u1, d1])
    u1 = layers.Conv2D(f, 3, padding='same', activation='relu')(u1)
    u1 = layers.Conv2D(f, 3, padding='same', activation='relu')(u1)
    
    # create an output layer
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u1)
    
    # combine input and output to create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # return the model
    return model