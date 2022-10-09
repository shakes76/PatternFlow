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
import matplotlib.pyplot as plt
import tensorflow as tf

class Dataset:
    """Represents and preprocesses the Facebook dataset"""
    def __init__(self, path, filename='facebook'):
        self.data_numpy = self._load(path, filename)
        self.data_tensor = self._tensify(self.data_numpy)

    def _load(self, path, filename):
        """Loads the partially preprocessed .npz Facebook dataset"""
        return np.load(path+"\\"+filename+'.npz')

    def _tensify(self, data_numpy):
        """
        Converts numpy arrays into tensors.
        """
        data = data_numpy

        edges = data["edges"].T
        features = tf.transpose(data["features"].T)
        weights = self._normalise_weights(data.get("weights"), edges)
        targets = data["target"].T

        print("adjacency shape:", edges.shape)
        print("features shape:", features.shape)
        print("weights shape:", weights.shape)
        print("targets shape:", targets.shape)

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

    def get_tensors(self):
        return self.data_tensor

    def get_edges(self):
        return self.data_tensor[0]

    def get_features(self):
        return self.data_tensor[1]

    def get_weights(self):
        return self.data_tensor[2]

    def get_targets(self):
        return self.data_tensor[3]

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