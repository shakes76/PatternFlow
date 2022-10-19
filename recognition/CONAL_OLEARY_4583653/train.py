import modules
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp


def trainVQVAE(vqvae, slice_train, slice_test):
    vqvae.fit(slice_train, epochs=10)
    vqvae.plot(10, slice_test)


def train(slice_train, slice_test):
    vqvae = modules.VQVAE()
    vqvae.compile(optimizer=keras.optimizers.Adam())
    print("Beginning training of VQVAE\n")
    trainVQVAE(vqvae, slice_train, slice_test)
