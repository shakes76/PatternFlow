"""
The training procedure to fit the GCN to the Facebook dataset.

Includes source code for training, validating, testing and saving the model. Also includes loss plots and metrics.

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""

from modules import GNN
from dataset import Dataset
import umap, umap.plot
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import keras

def loadClassifier(dataset, path):
    classifier = GNNClassifier(dataset, load_path=path)
    return classifier

class GNNClassifier():

    def __init__(self, data: Dataset, hidden_nodes=[32, 32], learning_rate=0.01, dropout_rate=0.2, load_path=None):
        self.num_classes = data.get_num_classes()
        self.data = data

        if load_path: # load old model
            self.model = self.load_model(load_path)
        else: # construct new model
            self.sample_graph = self.data
            self.model = GNN(self.sample_graph, self.num_classes, hidden_nodes, aggregation_type="sum", dropout_rate=dropout_rate)
            self._compile(learning_rate)

        self.model(self.data.get_training_split()[0]) # test model

    def _compile(self, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]

        # Compile the model.
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def _predictions_to_labels(self, predictions):
        """Returns the label predicted for each row of the predictions matrix.

        The predicted label is the index of the max value of the given row."""
        labels = []
        for prediction in predictions:
            labels.append(np.argmax(prediction))
        return labels

    def get_summary(self):
        self.model([1, 10, 100])
        self.model.summary()

    def train(self, epochs, batch_size):
        # Split data
        x_input, x_target = self.data.get_training_split()
        y_input, y_target = self.data.get_valid_split()

        # Create an early stopping callback
        #early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

        # Fit the model
        history = self.model.fit(x_input, x_target, epochs=epochs, batch_size=batch_size,
                                 validation_data=(y_input, y_target),
            #shuffle=True,
            #callbacks=[early_stopping],
        )

        return history

    def plot_curves(self, history):
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
        embeddings = self.model.get_node_embedding()
        targets = self.data.get_targets()
        mapper = umap.UMAP().fit(embeddings)

        umap.plot.points(mapper, labels=targets)
        umap.plot.connectivity(mapper, show_points=True)
        plt.show()

    def predict(self, dataset):
        """"Feed node indices into the given model. Compare predictions with known labels.

        Return raw predictions from model."""

        return self.model.predict(dataset[0])

    def predict_and_report(self, test_set=None, report_on=True):
        """Feed node indices into the given model. Compare predictions with known labels. Print report.

        Returns predicted labels."""
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

    def save(self, path='\\'):
        print("SAVING . . . ")
        self.model.save(path)
        print("SUCCESS!")
        return self.model

    def load_model(self, path):
        print("LOADING . . . ")
        model = keras.models.load_model(path)
        print("SUCCESS!")
        return model