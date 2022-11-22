"""
train.py
Functions used for training the Super-Resolution CNN. By executing this file the train function 
will run and start to train the model. 
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from modules import get_model, ESPCNCallback
from predict import displayPredictions
from dataset import *
from constants import EPOCHS

def train(epochs=EPOCHS):
    """Download the dataset and train the super-resolution CNN"""

    # Get datasets
    directory = downloadDataSet()
    train = getTraining(directory)
    valid = getValidation(directory)
    test = getTest(directory)

    # Get model
    model = get_model()
    model.summary()

    test_image = getTestImg(test)

    callbacks = [ESPCNCallback(test_image)]
    loss_fn = keras.losses.MeanSquaredError()
    optimizer = keras.optimizers.Adam(learning_rate=0.001)

    # Compile and train
    model.compile(
    optimizer=optimizer, loss=loss_fn,
    )

    history = model.fit(
        train, epochs=epochs, callbacks=callbacks, validation_data=valid, verbose=2
    )

    # Plot fit
    historyPlot(history)
    # Display some final predictions
    displayPredictions(model, test)
    

def historyPlot(model):
    """Plot training and validation loss of model after training"""
    plt.plot(model.history['loss'])
    plt.plot(model.history['val_loss'])
    plt.title('Super-resolution CNN loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper left')
    plt.show()

# Will run the model and consequently plot the results and prediction images
train()