import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import optimizers, losses
from tensorflow.keras.layers import Input, Layer, Dropout, Dot
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
    
    return features, labels, edges

def get_adj_matrix(labels, edges):
    """ Creates and adjacency matrix
    Parameters:
        labels: dataset of labels
        edges: dataset of edges
    Returns:
        An adjacency matrix
    """
    n = len(labels)
    A = np.eye(n, dtype = np.float32)
    for i in edges:
        A[i[0]][i[1]] = 1 

    A_tensorMatrix = tf.constant(A) 
    return A_tensorMatrix

def normalise_adj(adj_matrix):
    """
    Parameters:
        adj_matrix: adjacency matrix to be normalised, in form of tensor
    Returns:
        normalised adjacency matrix
    """
    # get inverse degree matrix
    total_neighbours = tf.math.reduce_sum(adj_matrix, 1)
    inv_deg_matrix = tf.linalg.diag(tf.math.reciprocal(total_neighbours))
    
    # get half 
    half_inv_deg_matrix = tf.math.sqrt(inv_deg_matrix)
    D_half = tf.constant(half_inv_deg_matrix)
    
    # multiply D*D*A
    A = tf.matmul(D_half, tf.matmul(D_half, adj_matrix))
    return A

def split_index(data):
    """ Partitions the dataset into training, validation, and testing splits
        of 0.2 : 0.2 : 0.6 since semi-supervised.
    Parameters:
        data: data to be split
    Returns:
        Indices of Training set, Validation, and Test set
    """
    size = int(len(data)*0.2)
    indices = [i for i in range(len(data))]

    # training split
    train_set = random.sample(indices, k = size)
    
    # split remainder of set
    remainder = set(indices).difference(train_set)

    val_set = random.sample(remainder, k = size)
    test_set = list(set(remainder).difference(val_set))
    
    return train_set, val_set, test_set

def GCN_Model(num_classes, num_channels = 16, dropout_rate = 0.5, kernel_regulariser = None, num_input_channels = None):
    """ Creates a GCN Model
    Parameters:
        num_classes: number of channels in output
        num_channels: number of channels in first GCN Layer
        dropout_rate: rate for Dropout Layers
        kernel_regulariser: regularisation applied to weights
        num_input_channels: number of input channels aka. node features
    """

    # Inputs
    x_input = Input((num_input_channels,), dtype = tf.float32)
    filter_input = Input((None,), dtype = tf.float32, sparse = True)

    # Create layers
    dropout_L0 = Dropout(dropout_rate)(x_input)
    gcn_L0 = GCN_Layer(num_channels, activation = "relu", kernel_regulariser= kernel_regulariser)([dropout_L0,filter_input])

    dropout_L1 = Dropout(dropout_rate)(gcn_L0)
    gcn_L1 = GCN_Layer(num_classes, acitvation = "softmax")([dropout_L1, filter_input])
    
    # Model
    model = Model(inputs = [x_input, filter_input], outputs = gcn_L1)

    return model

class GCN_Layer(Layer):
    """ A GCN layer.
    *Input*
    - Node features, with shape ([batch], num_nodes, num_features)
    *Output*
    - Node features
    Parameters:
        num_channels: number of output channels
        activation: activation function
        use_bias: boolean, whether to add a bias vector to output
        kernel_initialiser: intialiser for weights
        bias_initialiser: initialiser for bias vector
        kernel_regulariser: regularisation applied to weights
        bias_regulariser: regularisation applied to bias vector
        activity_regulariser: regularisation applied to output
    """
    def __init__(self, 
        num_channels, 
        activation = None, 
        use_bias = False, 
        kernel_initialiser = 'glorot_uniform',
        bias_initaliser = 'zeros',
        kernel_regulariser = None,
        bias_regulariser = None,
        activity_regulariser = None, **kwargs):

        super(GCN_Layer, self).__init__(**kwargs)
        self.num_channels = num_channels

    def build(self, input_shape): 
        assert len(input_shape)>= 2
        input_dim = input_shape[0][-1]

        # create weights of layer
        self.w = self.add_weight(shape = (input_dim, self.num_channels), 
            initializer= self.kernel_initialiser,
            name = "kernel",
            regularizer= self.kernel_regulariser)

    def call(self, inputs):
        x, a = inputs

        output = tf.matmul(x, self.w)
        output = tf.matmul(a, output)

        return self.activation(output)

    def config(self):
        return {"channels": self.num_channels}


#class GCN_Model(Model):
    """
    Graph Convolutional Network Model
    
    *INPUT*
    - Node features, with shape = ([batch], num_nodes, num_nodes)
    - Weighted adjacency matrix, with shape = ([batch], num_nodes, num_nodes)
    
    *OUTPUT*
    - Softmax predictions, with shape = ([batch], num_nodes, num_nodes)

    Parameters:
        num_labels: number of channels in output
        num_channels: number of channels in first GCN Layer
        dropout_rate: rate for Dropout Layers
        kernel_regulariser: regularisation applied to weights
        num_input_channels: number of input channels aka. node features

    """
    def __init__(self, num_labels, num_channels = 16, 
        dropout_rate = 0.5, kernel_regulariser = None, num_input_channels = None):
        
        super().__init__()

        self.num_labels = num_labels
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.kernel_regulariser = kernel_regulariser
        self.num_input_channels = num_input_channels

        # Inputs
        x_input = Input((num_input_channels,), dtype = tf.float32)
        filter_input = Input((None,), dtype = tf.float32, sparse = True)

        # Create layers
        self._dropoutL0 = Dropout(dropout_rate)(x_input)
        self._gcnL0 = GCN_Layer(num_channels, activation = "relu", 
            kernel_regulariser= kernel_regulariser)([self._dropoutL0,filter_input])

        self._dropoutL1 = Dropout(dropout_rate)(self._gcnL0)
        self._gcnL1 = GCN_Layer(num_labels, acitvation = "softmax")([self._dropoutL1, filter_input])
        
    def getconfig(self):
        return dict(
            num_labels = self.num_labels,
            channels = self.num_channels,
            activation = self.activation,
            output_act = self.output_activation,
            dropout_rate = self.dropout_rate,
            num_input_channels = self.num_input_channels,
        )

#if __name__ == '__main__':