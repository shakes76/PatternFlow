"""
The training procedure to fit the GCN to the Facebook dataset.

Includes source code for training, validating, testing and saving the model. Also includes loss plots and metrics.

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""

from modules import GNN
from dataset import Dataset
import umap, umap.plot
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras

def loadClassifier(dataset, path):
    """A helper function to build a GNN Classifier object that contains the loaded model."""
    classifier = GNNClassifier(dataset, load_path=path)
    return classifier

class GNNClassifier():
    """A wrapper that contains the GNN model, and implements training, validating, predicting, saving and loading of the
    model on the given dataset."""
    def __init__(self, data: Dataset, epochs=1, batch_size=256, save_path='.\\save', hidden_nodes=[32, 32], learning_rate=0.01, dropout_rate=0.2, load_path=None):
        self.num_classes = data.get_num_classes()
        self.data = data
        self.history = None

        if load_path: # load old model
            self.model = self.load(load_path)
        else: # construct, train and save new model
            self.model = GNN(self.data, self.num_classes, hidden_nodes, aggregation_type="sum", dropout_rate=dropout_rate)
            self._compile(learning_rate)
            self.history = self._train(epochs, batch_size)
            self.save(save_path)

        self.model(self.data.get_training_split()[0]) # ensure model is working

    def _compile(self, learning_rate):
        """Compile the model using default optimizer, loss and metric values."""
        # Initialise
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]

        # Compile the model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def _predictions_to_labels(self, predictions):
        """Returns the label predicted for each row of the predictions matrix.

        The predicted label is the index of the max value of the given row."""
        labels = []
        for prediction in predictions:
            # prediction is the index with the highest probability
            labels.append(np.argmax(prediction))
        return labels

    def get_summary(self):
        """Prints a summary of the model's architecture."""
        self.model.summary()

    def _train(self, epochs, batch_size):
        """Splits data into training and validation sets, and trains the model using the training set.

        Returns the accuracy and loss history of the training of the model."""
        # Split data
        x_input, x_target = self.data.get_training_split()
        y_input, y_target = self.data.get_valid_split()

        # Fit the model
        history = self.model.fit(x_input, x_target, epochs=epochs, batch_size=batch_size,
                                 validation_data=(y_input, y_target))
        return history

    def plot_curves(self):
        """
        Plots the loss and accuracy curves from the history of the model. Throws an error if model was pre-loaded and
        has not been trained since loading.
        """
        # get history
        if self.history:
            history = self.history
        else:
            assert "Model does not have any history."
            return

        # initialise plots
        fig, (acc, loss) = plt.subplots(1, 2, figsize=(15, 5))

        # plot Accuracy curve
        acc.plot(history.history['acc'], label='accuracy')
        acc.plot(history.history['val_acc'], label='val_accuracy')
        acc.set_xlabel('Epoch')
        acc.set_ylabel('Accuracy')
        acc.legend(["train", "test"], loc="lower right")

        # plot Loss curve
        loss.plot(history.history['loss'], label='loss')
        loss.plot(history.history['val_loss'], label='val_loss')
        loss.set_xlabel('Epoch')
        loss.set_ylabel('Loss')
        loss.legend(["train", "test"], loc="lower right")
        plt.show()

    def plot_umap(self):
        """
        Fits features data for each node to a 2-dimension UMAP manifold. Each point is coloured by ground truth label.

        Plots umap manifold.
        """
        # get data
        targets = self.data.get_targets()
        features = self.data.get_features()

        # transforms data to fit a umap manifold
        mapper = umap.UMAP().fit(features)

        # plot manifold
        umap.plot.points(mapper, labels=targets)
        umap.plot.connectivity(mapper, show_points=True)
        plt.show()

    def predict_and_report(self, test_set=None, report_on=True):
        """Feed node indices into the given model. Compare predictions with known labels.

        Prints report and returns predicted labels."""
        # test on pre-selected evaluation set
        if test_set:
            test_input, test_labels = test_set
        else:
            test_input = self.data.get_valid_split()[0]
            test_labels = self.data.get_valid_split()[1]

        predictions = self.model.predict(test_input) # outputs probability array
        predicted_labels = self._predictions_to_labels(predictions)  # process predictions
        correct = predicted_labels == test_labels
        total_test = len(test_input)

        if (report_on):
            print("Total Testing", total_test)
            # print("Gnd Truth:", test_labels)
            print("Predictions", predicted_labels)
            print("Which Correct:", correct)
            print("Total Correct:", np.sum(correct))
            print("Accuracy:", np.sum(correct) / total_test)

        return predicted_labels

    def save(self, path):
        """Saves the trained GNN model to a given path."""
        print("SAVING . . . ")
        self.model.save(path)
        print("SUCCESS!")
        return self.model

    def load(self, path):
        """Loads the trained GNN model from a given path."""
        print("LOADING . . . ")
        model = keras.models.load_model(path)
        print("SUCCESS!")
        return model