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

class GNNTrainer():

    def __init__(self, data: Dataset, hidden_nodes, learning_rate=0.01, dropout_rate=0.2):
        self.learning_rate = learning_rate
        self.num_classes = data.get_num_classes()
        self.data = data

        # Construct model
        self.sample_graph = self.data
        self.model = GNN(self.sample_graph, self.num_classes, hidden_nodes, aggregation_type="sum", dropout_rate=dropout_rate)
        self._compile()

    def _compile(self):
        optimizer = keras.optimizers.Adam(self.learning_rate)
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [keras.metrics.SparseCategoricalAccuracy(name="acc")]

        # Compile the model.
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def _split(self):
        seed = np.random.randint(1, 10000)
        data = self.data

        edges = data.get_edges()
        features = data.get_features()
        weights = data.get_weights()
        targets = data.get_targets()

        # Split nodes
        (self.x_train, self.x_valid,
         self.y_train, self.y_valid) = train_test_split(features, targets, test_size=0.33,
                                                        shuffle=True, random_state=seed)

        #print(data.get_data(), data.get_targets())
        #print(data.get_data().shape, data.get_targets().shape)

        # Split data
        #(self.x_train, self.x_valid,
        # self.y_train, self.y_valid) = train_test_split(data.get_data(), data.get_targets(), test_size=0.33,
        #                                                shuffle=True, random_state=seed)

        print(self.x_train, self.x_valid, self.y_train, self.y_valid)

    def get_summary(self):
        self.model([1, 10, 100])
        self.model.summary()

    def train(self, epochs, batch_size):
        # Create an early stopping callback
        #early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc", patience=50, restore_best_weights=True)
        #

        # Fit the model
        #print(self.data.get_data())
        print(self.data.get_data()[2])
        history = self.model.fit(self.data.get_data(), self.data.get_features(), epochs=epochs, batch_size=batch_size,
                                 validation_split=0.15,
                                 # shuffle=True,
                                 # callbacks=[early_stopping],
                                 )

        # Fit the model
        #history = self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
            #                      validation_split=0.15,
            #                     validation_data=(self.x_valid, self.y_valid),
            #shuffle=True,
            #callbacks=[early_stopping],
        #)

        return history

    def plot_curves(self, history):
        fig, (acc, loss) = plt.subplots(1, 2, figsize=(15, 5))

        # plot Accuracy curve
        acc.plot(history.history['accuracy'], label='accuracy')
        acc.plot(history.history['val_accuracy'], label='val_accuracy')
        acc.xlabel('Epoch')
        acc.ylabel('Accuracy')
        acc.legend(loc='lower right')

        # plot Loss curve
        loss.plot(history.history['loss'], label='loss')
        loss.plot(history.history['val_loss'], label='val_loss')
        loss.xlabel('Epoch')
        loss.ylabel('Loss')
        loss.legend(loc='lower right')
        plt.show()

    def save(self):
        pass

    def load(self):
        pass