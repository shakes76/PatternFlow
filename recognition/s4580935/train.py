from pickletools import optimize
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras import layers
import tensorflow_probability as tfp
import tensorflow as tf

def train(model, data, num_epochs):
    model.compile(optimizer=keras.optimizers.Adam())
    model.get_layer("encoder").summary()
    model.get_layer("decoder").summary()
    model.fit(data, epochs=num_epochs, batch_size=128, validation_split=0.2)