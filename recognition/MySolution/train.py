import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from predict import *


"""
Note: All the code used in this file has been inspired by
https://keras.io/examples/vision/image_classification_with_vision_transformer/
"""

"""
Class for training the model
"""
class Train:

    """
    Constructor for the Training class
    Initialises all the parameters to be used for training
    """
    def __init__(self, model):
        # parameters to be used for training
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.batch_size = 64
        self.num_epochs = 10
        self.model = model
        self.train(self.model)

    """
    Method used to train the model
    """
    def train(self, model):

        # Defines the optimizer to be used
        optimizer = tfa.optimizers.AdamW(
            learning_rate = self.learning_rate,
            weight_decay= self.weight_decay 
        )

        # Compiles the model
        model.compile(
            optimizer=optimizer,
            loss= tf.keras.losses.BinaryCrossentropy(from_logits=False), # Uses BinaryCrossentropy as there are only 2 classes
            metrics = [
                keras.metrics.BinaryAccuracy(name="Accuracy"),
            ],
        )

        # Trains the model
        history = model.fit(
            x=np.load('X_train.npy'),
            y=np.load('y_train.npy'),
            batch_size=self.batch_size,
            epochs=self.num_epochs,
            validation_split=0.1,
        )

        # Plots the results
        xs = range(self.num_epochs)
        plt.figure()
        plt.plot(xs, history.history["loss"], label="loss")
        plt.plot(xs, history.history["val_loss"], label="val_loss")
        plt.xlabel('Epoch')
        plt.ylabel('loss')
        plt.savefig("Loss.jpg")
        plt.imshow()
        plt.show()

        xs = range(self.num_epochs)
        plt.figure()
        plt.plot(xs, history.history["Accuracy"], label="Accuracy")
        plt.plot(xs, history.history["val_Accuracy"], label="val_Accuracy")
        plt.xlabel('Epoch')
        plt.ylabel('accuracy')
        plt.savefig("Accuracy.jpg")
        plt.imshow()
        plt.show()

        Predict(model)
        # test_loss, test_acc, *is_anything_else_being_returned = model.evaluate(np.load('X_test.npy'),  np.load('y_test.npy'),  verbose=2)
        # print(f"test_loss: {test_loss}")
        # print(f"test_acc: {test_acc}")
        return history