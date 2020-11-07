import tensorflow as tf
import matplotlib.pyplot as plt

import os
from PIL import Image

class KneeClassifier(object):
    """ A class which generates, compiles and learns weightings for a ConvNet to classify Knee Data """

    HEIGHT, WIDTH = 228, 260
    CHECKPOINT = './checkpoints/result'

    def __init__(self, relearn, training_data, testing_data):
        """ Constructs the KneeClassifier

        Parameters:
            relearn (bool): If we should relearn the weights from scratch
            training_data (tf.data.Dataset): The training data
            testing_data (tf.data.Dataset): The validation set
        """
        self._training_data = training_data
        self._testing_data = testing_data

        self._generate_model()
        self._compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

        # only relearn if specified, otherwise load weights from previous learning.
        if relearn:
            self._model.fit(
                training_data,
                epochs=40,
                shuffle=True,
                validation_data=testing_data,
                #callbacks=[tf.keras.callbacks.TensorBoard()] # never ended up using this
            )

        else:
            try:
                self._model.load_weights(self.CHECKPOINT)
            except:
                print("ERROR Loading Weights, Make sure you are calling the script from inside the same directory")


    def _generate_model(self):
        """ Generates the tensorflow model """
        self._model = tf.keras.Sequential([
        tf.keras.Input(shape=(self.HEIGHT, self.WIDTH, 1)),
        
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    def _compile(self, **kwargs):
        self._model.compile(**kwargs)

    def get_model(self):
        """ (tf.keras.Model) Return the generated model for some tutory tests"""
        return self._model

    def show_results(self, images_per_row, rows):
        """ Shows the result of some random classifications from the validation set 
        
        Parameters: 
            images_per_row (int): How many images to show per row
            rows (int): How many rows of images to show
        """
        for images, labels in self._testing_data.cache().repeat().take(1):
            # make some model predictions so we can see how our model fares
            predictions = self._model.predict(images[:images_per_row + 1])

            # added to make the image bigger
            plt.figure(num=None, figsize=(14, 8), dpi=80, facecolor='w', edgecolor='k')

            for i in range(1, len(predictions)):
                for j in range(rows):
                    ax = plt.subplot(3, len(predictions), i + j * len(predictions))
                    # plot the predicted and true labels of the images, 1 means right knee and 0 means left
                    ax.set_title(f"p: {round(predictions[i, 0], 1)}, t: {labels[i].numpy()}")
                    plt.imshow(tf.reshape(images[i + j * len(predictions)], (self.HEIGHT, self.WIDTH)))
                    plt.gray()
                    # remove ugly axes
                    ax.get_xaxis().set_visible(False)
                    ax.get_yaxis().set_visible(False)

    def save(self, path):
        self._model.save_weights(path)



