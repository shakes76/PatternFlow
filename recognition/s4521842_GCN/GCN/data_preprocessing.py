import tensorflow as tf
import numpy as np
import scipy.sparse as sp

class DataPreprocessing():
    def __init__(self, path='dataset/facebook.npz'):
        
        self.path = path
        self.data_edges = None
        self.data_features = None
        self.data_target = None
        self.n_nodes = None
        self.n_feature = None
        
        self.dataset = self.data_processing()
        
    
    def load_data(self):
        self.data = np.load(self.path, allow_pickle=True)
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
        print('-'*50)
        print('Number of nodes: ', self.n_nodes)
        print('Number of features of each node: ', self.n_feature)
        print('Categories of labels: ', set(self.data_target))
        
    def normalize_adjacency(self, adjacency):
        """calculate L=D^-0.5 * (A+I) * D^-0.5 """
        # Increase self-connection
        adjacency += sp.eye(adjacency.shape[0])
        degree = np.array(adjacency.sum(1))
        d_hat = sp.diags(np.power(degree, -0.5).flatten())

        return d_hat.dot(adjacency).dot(d_hat).tocsr().todense()
    
    def data_split(self, n_nodes):
        # split the dataset into train, validation and test set
        train_idx = range(int((n_nodes)*0.5))
        val_idx = range(int((n_nodes)*0.5), int((n_nodes)*0.75))
        test_idx = range(int((n_nodes)*0.75), n_nodes)
        
        train_mask = np.zeros(n_nodes, dtype=np.bool)
        val_mask = np.zeros(n_nodes, dtype=np.bool)
        test_mask = np.zeros(n_nodes, dtype=np.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True
        
        train_labels = self.data_target[train_idx]
        val_labels = self.data_target[val_idx]
        test_labels = self.data_target[test_idx]

        return train_mask, val_mask, test_mask, train_labels, val_labels, test_labels
    
    def one_hot_encoding(self, labels):
        
        label_encoder = LabelBinarizer()
        encoded_labels = label_encoder.fit_transform(labels)
        return tf.convert_to_tensor(encoded_labels, dtype=tf.float32)
        
    def data_processing(self):
        
        self.load_data()
        
        onehot_labels = self.one_hot_encoding(self.data_target)
        
        # build graph
        # construct sparse matrix in COOrdinate format.
        adjacency = sp.coo_matrix(
            (np.ones(self.data_edges.shape[0]), (self.data_edges[:, 0], self.data_edges[:, 1])), 
            shape=(self.data_target.shape[0], self.data_target.shape[0]),
            dtype=np.float32
        )
        
        # normalize adjacency
        adjacency_norm = self.normalize_adjacency(adjacency)
        
        train_mask, val_mask, test_mask, train_labels, val_labels, test_labels= self.data_split(self.n_nodes)
        
        train_labels = self.one_hot_encoding(train_labels)
        val_labels = self.one_hot_encoding(val_labels)
        test_labels = self.one_hot_encoding(test_labels)
        
        return self.data_features, onehot_labels, adjacency_norm, train_mask, val_mask, test_mask, train_labels, val_labels, test_labels
        
    def get_data(self):
        return self.dataset
    
    def get_test_labels(self):
        _, _, _, _, _, test_labels = self.data_split(self.n_nodes)
        
        return test_labels
        
    
