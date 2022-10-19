"""
dataset.py containing the data loader for loading and preprocessing data
Reference: https://github.com/ElonQuasimodoYoung/COMP3710_Report_Donghao_Yang_45930032_PatternFlow/blob/topic-recognition/recognition/s4521842_GCN/GCN/data_preprocessing.py
"""

# All needed library for loading and preprocessing data
import numpy as np
import scipy.sparse as sp
import sklearn.preprocessing as preprocessing
import sklearn.model_selection as model_selection


def normalize_adjacency_matrix(adjacency_matrix):
    """
    This function makes use of the equation 'L=D^(-0.5)*(A+I)*D^(-0.5)' to normalize the adjacency matrix in
    case it's over scale along with the calculation of GCN. The D is diagonal matrix where the numbers in the
    diagonal are the degrees of every node in the graph. A is the adjacency matrix, and I is identity matrix.
    """
    adjacency_matrix = sp.eye(adjacency_matrix.shape[0]) + adjacency_matrix
    degree_array = np.array(adjacency_matrix.sum(1))  # Sum the matrix elements over a given axis.
    degree_diagonal_matrix = sp.diags(np.power(degree_array, -0.5).flatten())
    adjacency_matrix = degree_diagonal_matrix.dot(adjacency_matrix).dot(degree_diagonal_matrix).tocsr()
    return adjacency_matrix


def load_facebook_page_data(path='./facebook.npz'):
    """This function loads Facebook Large Page-Page Network dataset """
    # load features, edges and targets date respectively.
    data = np.load(path)
    features = data['features']
    edges = data['edges']
    targets = data['target']
    print('Features data shape', features.shape)
    print('Edges data shape', edges.shape)
    print('Target data shape', targets.shape)
    nodes_number = features.shape[0]
    features_number = features.shape[1]
    print('The number of nodes', nodes_number, '; The number of features', features_number)

    # Create the adjacency matrix. Since the density of our dataset is only 0.001, we transfer it to
    # Compressed Sparse Row matrix
    adjacency_matrix = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                                     shape=(nodes_number, nodes_number), dtype=np.float32)
    adjacency_matrix = sp.csr_matrix(adjacency_matrix)
    adjacency_matrix = normalize_adjacency_matrix(adjacency_matrix)

    # Create the feature matrix and normalize it.
    features_matrix = preprocessing.normalize(features)

    # Convert multi-class labels to binary labels. (One hot encoding) and split the data set to
    # train, validation and test respectively.
    targets = preprocessing.LabelBinarizer().fit_transform(targets)
    train_data_range = range(int(nodes_number * 0.5))
    validation_data_range = range(int(nodes_number * 0.5), int(nodes_number * 0.75))
    test_data_range = range(int(nodes_number * 0.75), nodes_number)
    train_data_mask = np.zeros(nodes_number, dtype=bool)
    validation_data_mask = np.zeros(nodes_number, dtype=bool)
    test_data_mask = np.zeros(nodes_number, dtype=bool)
    train_data_mask[train_data_range] = True
    validation_data_mask[validation_data_range] = True
    test_data_mask[test_data_range] = True
    # split targets into train, validation and test in terms of ratio(0.50, 0.25, 0.25) respectively
    train_target, validation_test_target = model_selection.train_test_split(targets, train_size=0.5,
                                                                            shuffle=False)
    validation_target, test_target = model_selection.train_test_split(validation_test_target, train_size=0.5,
                                                                      shuffle=False)
    return adjacency_matrix, features_matrix, targets, train_target, validation_target, test_target, train_data_mask, \
           validation_data_mask, test_data_mask
