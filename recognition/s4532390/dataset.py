import numpy as np
import tensorflow as tf
from sklearn import model_selection
from sklearn import preprocessing
import scipy.sparse

def parse_data(file_path, test_size):
    """
    
    """
    # Extracts the data from the npz file
    with np.load(file_path) as data:

        # 22470 integers (0, 1, 2, 3) corresponding to the categories - politicians, governmental organizations, television shows and companies
        targets = data['target']

        # 22470 vectors - each with 128 features
        features = data['features']

        # 342004 size edge list
        edges = data['edges']

        # Constructing adjaceny matrix
        num_pages = len(features)
        num_edges = len(edges)
        edge_in = edges[:,0]
        edge_out = edges[:,1]

        adjacency_matrix = scipy.sparse.coo_matrix((np.ones(num_edges), (edge_in, edge_out)),
                                                    shape=(num_pages, num_pages),
                                                    dtype=np.float32)

        # Add identity matrix as each page connects to itself
        adjacency_matrix += scipy.sparse.eye(adjacency_matrix.shape[0])
        # Normalise the matrix
        adjacency_matrix = preprocessing.normalize(adjacency_matrix, axis=1, norm='l1')
        
        X_train, X_test, y_train, y_test = model_selection.train_test_split(features, targets, test_size=test_size)
        
        X_train = tf.convert_to_tensor(X_train)
        X_test = tf.convert_to_tensor(X_test)
        y_train = tf.convert_to_tensor(y_train)
        y_test = tf.convert_to_tensor(y_test)



parse_data('recognition\\s4532390\\res\\facebook.npz', 0.2)

    



    # data[]
# data = np.load('facebook.npz')

# print(data)