"""
Trains perceiver transformer on the dataset

@author: Pritish Roy
@email: pritish.roy@uq.edu.au
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from perceiver.load_data import LoadDataset
from perceiver.perceiver import PerceiverTransformer
from settings.config import *


class TransformerTrainer:
    """Responsible for training the perceiver transformer model"""

    def __init__(self):
        """constructor for initialising the dataset and Perceiver model"""
        self.data = LoadDataset()
        self.model = PerceiverTransformer(self.data)

        self.history = None

    @staticmethod
    def scheduler(epoch, learning_rate):
        """reduce the learning rate in each epoch"""
        return learning_rate if epoch < 2 else learning_rate * tf.math.exp(-0.1)

    def train_perceiver_transformer(self):
        """ This methods does the following:
        compiles the model,
        training using keras fit,
        outputs the model's summary,
        plot the model layers,
        saves the model weights,
        finally print's the test accuracy."""
        # compile the model
        self.model.compile(
            optimizer=tf.optimizers.SGD(
                learning_rate=LEARNING_RATE,
                momentum=MOMENTUM,
                nesterov=False),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

        # fit model
        self.history = \
            self.model.fit(x=self.data.x_train,
                           y=self.data.y_train,
                           batch_size=BATCH_SIZE,
                           epochs=EPOCHS,
                           steps_per_epoch=int(np.ceil(
                               self.data.x_train.shape[
                                   0] / BATCH_SIZE)),
                           validation_split=VALIDATION_SPLIT,
                           shuffle=True,
                           callbacks=[
                               tf.keras.callbacks.EarlyStopping(
                                   monitor=MONITOR,
                                   patience=PATIENCE,
                                   restore_best_weights=True
                               ),
                               tf.keras.callbacks.LearningRateScheduler(
                                   self.scheduler)])

        self.model.summary()

        # plot model
        # todo: fix
        tf.keras.utils.plot_model(
            self.model, to_file=f'{FIGURE_LOCATION}perceiver_transformer.png',
            show_shapes=True, show_dtype=True,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
        )

        # save model
        self.model.save_weights(f'{LOGS_PATH}perceiver_transformer.h5')

        # test evaluation
        _, accuracy = self.model.evaluate(self.data.x_test, self.data.y_test)
        print(f"Test accuracy: {round(accuracy * 100, 2)}%")

    def plot_graph(self):
        """plot graph for models loss and accuracy from model history"""
        for graph in GRAPHS_PLOT:
            # initialise pyplot figure
            fig = plt.figure()

            # plot
            plt.plot(self.history.history[f'{graph}'])
            plt.plot(self.history.history[f'val_{graph}'])

            # plot graph title
            plt.title(f'model-{graph}')

            # plot graph label's
            plt.ylabel(PLOT_Y_LABEL)
            plt.xlabel(graph)
            plt.legend(LEGEND)

            # save figure
            fig.savefig(f'{FIGURE_LOCATION}{graph}.png')

    def do_action(self):
        """sequential set of actions"""
        self.train_perceiver_transformer()
        self.plot_graph()
