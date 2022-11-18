"""
The data loader for loading and pre-processing the Facebook Large Page-Page Network.

The loaded dataset should be the partially pre-processed dataset in .NPZ format ("facebook.NPZ").

Labels:
0 = tvshow
1 = company
2 = government
3 = politician

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

class Dataset:
    """Represents and preprocesses the Facebook dataset"""
    def __init__(self, path, test_size=0.33):
        self.seed = np.random.randint(1, 10000)
        self.data_numpy = np.load(path)
        self.num_classes = len(set(self.data_numpy["target"]))
        self.data_tensor = self._tensify(self.data_numpy)
        self.split = self._split(test_size) # split is immutable

    def _tensify(self, data_numpy):
        """
        Converts numpy arrays into tensors.
        """
        data = data_numpy

        edges = tf.transpose(tf.convert_to_tensor(data["edges"]))
        features = tf.transpose(data["features"].T)
        weights = self._normalise_weights(data.get("weights"), edges)
        targets = tf.convert_to_tensor(data["target"])

        return edges, features, weights, targets

    def _normalise_weights(self, weights, edges):
        """
        Normalise weights. If no weights are provided, create an identity matrix of weights.
        """
        if weights is None:
            weights = tf.ones(shape=edges.shape[1])
        else:
            weights.T
        # Force weights to sum to 1.
        return weights / tf.math.reduce_sum(weights)

    def _split(self, test_size):
        """
        Creates a training and validation split for the dataset. Returns a tensor.

        The data being split are node ids and targets. Node ids will be given to the model and the model will train and
        predict targets for the given ids.

        :params test_size: proportion of data to be used in the test set
        """
        ids = self.get_ids().numpy()
        targets = self.get_targets().numpy()

        # Split nodes
        train_x, labels_x, train_y, labels_y = train_test_split(ids, targets, test_size=test_size, shuffle=True, random_state=self.seed)

        return (tf.convert_to_tensor(train_x), tf.convert_to_tensor(labels_x),
               tf.convert_to_tensor(train_y), tf.convert_to_tensor(labels_y))

    def get_training_split(self):
        """Get predetermined training split. Returns tensors (ids, labels)."""
        return self.split[0], self.split[2]

    def get_valid_split(self):
        """Get predetermined validation split. Returns tensors (ids, labels)."""
        return self.split[1], self.split[3]

    def get_data(self):
        """Returns tensors (features, edges, edge_weights)."""
        return self.get_features(), self.get_edges(), self.get_weights()

    def get_edges(self):
        """Returns tensor of all edges."""
        return self.data_tensor[0]

    def get_features(self):
        """Returns tensor of features for each node."""
        return self.data_tensor[1]

    def get_weights(self):
        """Returns tensor of all edge weights."""
        return self.data_tensor[2]

    def get_ids(self):
        """Returns tensor of all node ids."""
        # generate a tensor of ids
        return tf.range(0, self.get_features().shape[0])

    def get_targets(self):
        """Returns tensor of labels (i.e. targets)."""
        return self.data_tensor[3]

    def get_num_classes(self):
        """Returns number of classes of labels (i.e. 4)."""
        return self.num_classes

    def summary(self, n=5):
        """
        Prints a summary of the .npz dataset

        :params n: number of data points to print
        """
        data = self.data_numpy

        # Print n example edges
        print("\nEXAMPLE EDGES")
        for i in range(0, n):
            print("Edges:", data["edges"][i])

        # Print n example nodes
        print("\nEXAMPLE NODES")
        for i in range(0, n):
            print("\nNode", i,
                  "\nFeatures:", data["features"][i],
                  "\nTarget:", data["target"][i])

        # Summary of dataset
        print("\nSUMMARY",
              "\nNum of Edges:", len(data["edges"]), "/ 2",
              "\nNum of Nodes", len(data["features"][0]),
              "\nNum of Targets:", 4)
        for i in range(0, n):
            print("Node:", i,
                  "Num of Features:", len(data["features"][i]),
                  "Target:", data["target"][i])