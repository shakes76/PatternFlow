"""
The training procedure to fit the GCN to the Facebook dataset.

Includes source code for training, validating, testing and saving the model. Also includes loss plots and metrics.

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""

from modules import GNN
from dataset import Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
import numpy as np

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

    def _compile(self, learning_rate):
        optimizer = keras.optimizers.Adam(learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")]

        # Compile the model.
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def _split(self):
        seed = np.random.randint(1, 10000)
        ids = self.data.get_ids().numpy()
        targets = self.data.get_targets().numpy()

        print("ids", ids.shape, ids)
        print("targets", targets.shape, targets)

        # Split nodes
        (self.x_train, self.x_valid,
         self.y_train, self.y_valid) = train_test_split(ids, targets, test_size=0.33,
                                                        shuffle=True, random_state=seed)

    def get_summary(self):
        self.model([1, 10, 100])
        self.model.summary()

    def train(self, epochs, batch_size):
        # Split data
        self._split()

        # Create an early stopping callback
        #early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)

        # Fit the model
        history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=(self.x_valid, self.y_valid),
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

    # def predict_and_report(self, test_data, test_labels, model, report_on=True):
    #     """Feed test_images into the given model. Compare predictions with test_labels.
    #
    #     Returns predicted_labels."""
    #
    #     predictions = model.predict(test_images)  # outputs probability array
    #     predicted_labels = predictions_to_labels(predictions)  # process predictions
    #     correct = predicted_labels == test_labels
    #     total_test = len(test_images)
    #
    #     if (report_on):
    #         print("Total Testing", total_test)
    #         # print("Gnd Truth:", test_labels)
    #         print("Predictions", predicted_labels)
    #         print("Which Correct:", correct)
    #         print("Total Correct:", np.sum(correct))
    #         print("Accuracy:", np.sum(correct) / total_test)
    #
    #     return predicted_labels

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