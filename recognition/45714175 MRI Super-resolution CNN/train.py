"""
train.py
Used for training the model
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from modules import get_model, ESPCNCallback
from predict import predict
from dataset import *

def train(epochs=30):
    """Download the dataset and train the super-resolution CNN"""

    # Get datasets
    directory = downloadDataSet()
    train = getTraining(directory)
    valid = getValidation(directory)
    test = getTest(directory)

    # Get model
    model = get_model()
    model.summary()

    test_image = Null

    callbacks = [ESPCNCallback(test_image)]
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    #
    model.compile(
    optimizer=optimizer, loss=loss_fn,
    )

    history = model.fit(
        train, epochs=epochs, callbacks=callbacks, validation_data=valid, verbose=2
    )