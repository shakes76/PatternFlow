import numpy as np
from sklearn import preprocessing
import torch
import scipy.sparse
import torch.nn.functional as F

def parse_data(file_path):
    """
    Parses data from facebook.npz
    Returns:
        features - 128 dim vectors (tensor)
        adjacency matrix - adjacency matrix of pages (sparse tensor)
        targets - class for each page (tensor)
    """
    print("Parsing data...")
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

        # Constructing edge matrix
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
        coo_matrix = preprocessing.normalize(coo_matrix, axis=1, norm='l1')
        coo_matrix = coo_matrix.tocoo().astype(np.float32)

        # Getting values needed for the adjacency matrix
        indices = torch.from_numpy(np.vstack((coo_matrix.row, coo_matrix.col)).astype(np.int64))
        values = torch.from_numpy(coo_matrix.data)
        shape = torch.Size(coo_matrix.shape)
        
        # Creating the adjacency matrix
        adjacency_matrix = torch.sparse_coo_tensor(indices, values, shape)

        return features, adjacency_matrix, targets