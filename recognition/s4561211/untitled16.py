# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 21:31:12 2021

@author: yayu kang
"""

import numpy as np
import scipy.sparse as sp
import torch 

def load_data(path):
  """
      Load data data and transform adjacency matrix

      Parameters:
          path (str): the path of data
          
      Returns:
          (scipy.sparse.coo.coo_matrix): Return adjacency matrix
          (scipy.sparse.csr.csr_matrix): Return features
          (torch.Tensor): Return labels
  """
  data = np.load(path)
  edges = data['edges']
  edges = np.unique(edges, axis=0)
  features = data['features']
  labels = data['target']

  features = sp.csr_matrix(features)
  adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),shape=(labels.shape[0], labels.shape[0]))#.toarray()

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

