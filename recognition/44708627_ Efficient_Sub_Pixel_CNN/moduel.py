#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import Orthogonal


# In[ ]:


def res_block(inputs, numLayers):

    channels = inputs.get_shape()[-1]
    storedOutputs = [inputs]
    
    # add layers
    for _ in range(numLayers):
        
        # concatenate the previous outputs and pass it through a
        # CONV layer, and append the output to the ongoing concatenation
        
        localConcat = tf.concat(storedOutputs, axis=-1)
        out = layers.Conv2D(filters=channels, kernel_size=3, padding="same",
            activation="relu",
            kernel_initializer="Orthogonal")(localConcat)
        storedOutputs.append(out)
        
    # concatenate all the outputs, pass it through a pointwise
    # convolutional layer, and add the outputs to initial inputs
    finalConcat = tf.concat(storedOutputs, axis=-1)
    finalOut = layers.Conv2D(filters=inputs.get_shape()[-1], kernel_size=1,
        padding="same", activation="relu",
        kernel_initializer="Orthogonal")(finalConcat)
    finalOut = layers.Add()([finalOut, inputs])
    
    return finalOut


# In[4]:


def get_model(upscale_factor=4, channels=1):
    conv_args = {
        "activation": "relu",
        "kernel_initializer": "Orthogonal",
        "padding": "same",
    }
    inputs = keras.Input(shape=(None, None, channels))
    x = layers.Conv2D(64, 5, **conv_args)(inputs)
    x = layers.Conv2D(64, 3, **conv_args)(x)
    x = res_block(x, 3)
    x = layers.Conv2D(32, 3, **conv_args)(x)
    x = res_block(x, 3)
    x = layers.Conv2D(channels * (upscale_factor ** 2), 3, **conv_args)(x)
    outputs = tf.nn.depth_to_space(x, upscale_factor)

    return keras.Model(inputs, outputs)

