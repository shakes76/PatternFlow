'''
Author name: Arsh Upadhyaya, s47539934
To preprocess dataset facebook.npz
'''

import numpy as np
from sklearn import preprocessing
import torch
import scipy.sparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch.optim as optim


def load_data(file_path):
    '''
    parameters: 
    takes file path
    returns:
    Adjacency matrix: normalized coo matrix
    features and labels(targets) as tensors
    '''

    data = np.load("/content/drive/MyDrive/facebook.npz")
    edges = data['edges']
    features = data['features']
    labels = data['target']

    features = sp.csr_matrix(features)

    adj= sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]))

    #normalize
    colsum = np.array(adj.sum(0))
    D = np.power(colsum, -1)[0]
    D[np.isinf(D)] = 0
    D_inv = sp.diags(D)
    adj_trans = D_inv.dot(adj)

    #transform data type
    indices = torch.LongTensor(np.vstack((adj_trans.tocoo().row, adj_trans.tocoo().col)))
    values = torch.FloatTensor(adj_trans.data)
    shape = adj_trans.shape

    adj_trans = torch.sparse_coo_tensor(indices, values, shape)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    return adj_trans, features, labels
   
