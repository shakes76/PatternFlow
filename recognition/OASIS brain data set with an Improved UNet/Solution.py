"""
Unet Model

@author Max Hornigold
"""


import tensorflow as tf
from tensorflow.keras import layers

def unet_model(output_channels, f=6):
    """Creates and returns a UNET model"""
    
    inputs = tf.keras.layers.Input(shape=(256, 256, 1))
    
    # Downsampling through the model
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
    
    # Upsampling and establishing the skip connections
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
    
    # This is the last layer of the model.
    outputs = layers.Conv2D(output_channels, 1, activation='softmax')(u1)
    
    # Create model using the layers
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # return the model
    return model