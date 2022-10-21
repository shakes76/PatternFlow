import tensorflow as tf
import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import LabelBinarizer, normalize


def normalize_adjacency(adjacency):
    """calculate L=D^-0.5 * (A+I) * D^-0.5 """
    # Increase self-connection
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocsr().todense()


def one_hot_encoding(labels):
    """
        convert each categorical value into a new categorical column
        and assign a binary value of 1 or 0 to those columns
    """
    label_encoder = LabelBinarizer()
    encoded_labels = label_encoder.fit_transform(labels)
    return tf.convert_to_tensor(encoded_labels, dtype=tf.float32)


class DataPreprocessing:
    def __init__(self, path='dataset/facebook.npz'):
        self.data = np.load(path, allow_pickle=True)
        self.data_edges = None
        self.data_features = None
        self.data_target = None
        self.n_nodes = None
        self.n_feature = None

        self.dataset = self.data_processing()

    def load_data(self):
        self.data_edges = self.data['edges']
        self.data_features = self.data['features']
        self.data_target = self.data['target']

        # the number of nodes
        self.n_nodes = self.data_features.shape[0]
        # the size of node features
        self.n_feature = self.data_features.shape[1]

        print('Shape of Edge data', self.data_edges.shape)
        print('Shape of Feature data', self.data_features.shape)
        print('Shape of Target data', self.data_target.shape)
        print('-' * 50)
        print('Number of nodes: ', self.n_nodes)
        print('Number of features of each node: ', self.n_feature)
        print('Categories of labels: ', set(self.data_target))

    def data_split(self, n_nodes):
        """
            Split dataset into training, validation, test set.
            training_set : validation_set : test_set = 0.3:0.2:0.5
        """
        # get the index of each dataset
        train_idx = range(int((n_nodes) * 0.3))
        val_idx = range(int((n_nodes) * 0.3), int((n_nodes) * 0.5))
        test_idx = range(int((n_nodes) * 0.5), n_nodes)

        # set the mask
        train_mask = np.zeros(n_nodes, dtype=np.bool)
        val_mask = np.zeros(n_nodes, dtype=np.bool)
        test_mask = np.zeros(n_nodes, dtype=np.bool)
        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

        # split labels into training, validation, test set
        train_labels = self.data_target[train_idx]
        val_labels = self.data_target[val_idx]
        test_labels = self.data_target[test_idx]

        return train_mask, val_mask, test_mask, train_labels, val_labels, test_labels

    def data_processing(self):
        # load data
        self.load_data()
        # one hot encoding
        one_hot_labels = one_hot_encoding(self.data_target)
        # normalize feature
        self.data_features = normalize(self.data_features)

        # build graph
        # construct sparse matrix in COOrdinate format.
        adjacency = sp.coo_matrix(
            (np.ones(self.data_edges.shape[0]), (self.data_edges[:, 0], self.data_edges[:, 1])),
            shape=(self.data_target.shape[0], self.data_target.shape[0]),
            dtype=np.float32
        )

        # normalize adjacency
        adjacency_norm = normalize_adjacency(adjacency)
        # split dataset
        train_mask, val_mask, test_mask, train_labels, val_labels, test_labels = self.data_split(self.n_nodes)
        # one hot encoding for label sets
        train_labels = one_hot_encoding(train_labels)
        val_labels = one_hot_encoding(val_labels)
        test_labels = one_hot_encoding(test_labels)

        return self.data_features, one_hot_labels, adjacency_norm, train_mask, val_mask, test_mask, train_labels, val_labels, test_labels

    def get_data(self):
        return self.dataset

    def get_test_labels(self):
        _, _, _, _, _, test_labels = self.data_split(self.n_nodes)
        return test_labels
