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

model.summary()