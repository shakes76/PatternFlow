import tensorflow as tf
from keras import layers, models
import numpy as np
import torch

model = models.Sequential([
    
    layers.Conv2D(32, 3, padding='same', activation='relu',  input_shape=(32, 32, 3)),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Conv2D(128, 3, padding="same", activation="relu")
])


def down_sample_block(x, n_filters):
    f = model(x, n_filters)
    p = layers.MaxPool2D(2, 2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


def up_sample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = model(x, n_filters)

    return x

model.summary()