import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import Input, Layer, Dense, Dropout
from tensorflow.keras.models import Model

from sklearn.manifold import TSNE

def load_data():
    """  Loads the preprocessed data from provided facebook.npz file.
    
    Returns:
        Adjacency matrix, features, and labels
    """
    data = np.load('facebook.npz')
    
    edges = data['edges']
    labels = data['target']
    features = data['features']
    
    # adjacency matrix
    n = len(labels)
    A = np.eye(n, dtype = np.float32)
    for i in edges:
        A[i[0]][i[1]] = 1 
    
    # check loaded properly
    print("num of classes: " + str(len(np.unique(labels))))
    print("num of nodes: " + str(features.shape[0]))
    print("num of edges: "+ str(len(edges)/2))
    print("num of features: " + str(features.shape[1]))

    A_tensorMatrix = tf.constant(A)

class GCN(Layer):
    """ Graph Convolution Layer

    """

    def __init__(self, ):
        super(GCN, self).__init__()

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs):
        pass


class GCN_Model(Model):
    """
    
    """

    def __init__(self, num_features, num_nodes, num_classes):
        """
        Creates a Graph Convolutional Network Model
        Parameters:
            num_features: number of features
            num_nodes: number of nodes
            num_classes: number of classes
            channels1: number of channels in first layer
            channels2: number of channels in second layer
            dropout: dropout rate

        Returns:
            Multi-layer GCN Model
        """
        super(GCN_Model, self).__init__()
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.num_classes = num_classes


        self.dense = Dense(4, activation = tf.nn.relu)
        self.dense2 = Dense(5, activation= tf.nn.softmax)

        self.dropout = Dropout(0.5)


def train(num_epochs):
    """
    Train model

    Parameters:
        num_epochs: number of epochs ie. iterations
    
    Returns:

    
    """

    for epoch in range(num_epochs):

        # Train





        # Validate


        # Save






if __name__ == '__main__':
    # Load Data
    load_data()

    # Semi-supervised split -> 20:20:60


    # Create GCN

    # Train

    # Validate

    # Test

    # Plot TSNE
