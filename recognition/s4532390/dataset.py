import numpy as np
from sklearn import preprocessing
import torch
import scipy.sparse
import torch.nn.functional as F

def parse_data(file_path):
    """
    
    """
    # Extracts the data from the npz file
    with np.load(file_path) as data:

        # 22470 integers (0, 1, 2, 3) corresponding to the categories - politicians, governmental organizations, television shows and companies
        targets = data['target']
        targets = torch.from_numpy(targets)

        # 22470 vectors - each with 128 features
        features = data['features']
        features = torch.from_numpy(features)
        features = F.normalize(features, p=1, dim=1)

        # 342004 size edge list
        edges = data['edges']

        # Constructing adjaceny matrix
        num_pages = len(features)
        num_edges = len(edges)
        edge_in = edges[:,0]
        edge_out = edges[:,1]

        coo_matrix = scipy.sparse.coo_matrix((np.ones(num_edges), (edge_in, edge_out)),
                                                    shape=(num_pages, num_pages),
                                                    dtype=np.float32)
        # Add identity matrix as each page connects to itself
        coo_matrix += scipy.sparse.eye(coo_matrix.shape[0])
        # Normalise the matrix
        # coo_matrix = F.normalize(coo_matrix, p=1, dim=1)
        coo_matrix = preprocessing.normalize(coo_matrix, axis=1, norm='l1')

        # sparse = coo_matrix.tocoo().astype(np.float32)

        # Convert to sparse tensor TODO CHANGE THIS CODE
        def convert_to_adj_tensor(input):

            input = input.tocoo().astype(np.float32)

            indices = torch.from_numpy(np.vstack((input.row, input.col)).astype(np.int64))
            values = torch.from_numpy(input.data)
            shape = torch.Size(input.shape)
            return torch.sparse_coo_tensor(indices, values, shape)

        adjacency_matrix = convert_to_adj_tensor(coo_matrix)

        return features, adjacency_matrix, targets

# What data shape does it want? How to use the 128 dim features?
# How to split dataset if it is a single graph?
# Why make a custom GCNConv layer? Use custom
# Embeddings? See shakes code
# Semi-supervised? Fully is fine