    """
    code for testing the algorithm
    by: Kexin Peng, 4659241
    """
import numpy as np
import scipy.sparse as sp
from algorithm import *

# load data
facebook = np.load('facebook.npz')
edges = facebook['edges']
features = facebook['features']
target = facebook['target']

num_classes = len(np.unique(target))
num_nodes = features.shape[0]
num_features = features.shape[1]
num_edges = edges.shape[0]

def normalize(matrix):
    row_sum = np.array(matrix.sum(1))
    # 1/sum
    sum_inv = np.power(row_sum, -1).flatten() 
    # if it's infinite, change to 0
    sum_inv[np.isinf(sum_inv)] = 0.   
    # build diagonal matrix
    diag_mat = sp.diags(sum_inv)
    # normalized matrix: D^-1 * matrix
    normalized = diag_mat.dot(matrix) 
    return normalized

# normalize features and transform data to tensor
norm_features = normalize(features)
norm_features = torch.from_numpy(norm_features)
target = torch.from_numpy(target)

# Adjacency matrix A-- n*n
edge_data = np.ones(num_edges)
row = edges[:, 0]
col = edges[:, 1]
adj_matrix = sp.coo_matrix((edge_data, (row, col)),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)
# A+I add identity matrix
new_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
# normalize the matrix
new_matrix = normalize(new_matrix) 