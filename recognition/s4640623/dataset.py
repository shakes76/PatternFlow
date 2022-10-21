import numpy as np
import torch
import scipy.sparse as sp
from io import BytesIO
import requests

train_size = 3 # Number to divide dataset by for train, valid and test datasets

"""Downloads data from URL, 
default is preprocessed Facebook Large Page-Page Network dataset"""
def load_data(path="https://graphmining.ai/datasets/ptg/facebook.npz"):
  r = requests.get(path, stream = True)
  data = np.load(BytesIO(r.raw.read()))
  return data

"""Create training, validating, and test arrays to segment the dataset
Convert features, target and edges from dataset to PyTorch tensors"""
def process_data(dataset):
  train_ds = range(0,round(dataset['features'].shape[0]/3))
  valid_ds = range(round(dataset['features'].shape[0]/3), 
                  round(2*dataset['features'].shape[0]/3))
  test_ds = range(round(2*dataset['features'].shape[0]/3), 
                  round(dataset['features'].shape[0]))

  train_ds = torch.LongTensor(train_ds)
  valid_ds = torch.LongTensor(valid_ds)
  test_ds = torch.LongTensor(test_ds)

  features = torch.as_tensor(dataset['features'])
  target = torch.from_numpy(dataset['target'])
  edges = torch.from_numpy(dataset['edges'])

  return train_ds, valid_ds, test_ds, features, target, edges

"""Normalise a given matrix"""
def normalise(mx):
  r_inv = np.power(np.array(mx.sum(1)), -1).flatten()
  r_inv[np.isinf(r_inv)] = 0.
  r_mat_inv = sp.diags(r_inv)
  mx = r_mat_inv.dot(mx)
  return mx

"""Convert a given sparse matrix to sparse tensor"""
def sparse_mx_to_sparse_tensor(mx):
  mx = mx.tocoo().astype(np.float32)
  indices = torch.from_numpy(
      np.vstack((mx.row, mx.col)).astype(np.int64))
  values = torch.from_numpy(mx.data)
  shape = torch.Size(mx.shape)
  return torch.sparse.FloatTensor(indices, values, shape)

"""Create adjacency matrix"""
def adj_matrix(dataset):
  adj = sp.coo_matrix((np.ones(dataset['edges'].shape[0]), 
                       (dataset['edges'][0:dataset['edges'].shape[0], 0], 
                        dataset['edges'][0:dataset['edges'].shape[0], 1])),
                      shape=(dataset['target'].shape[0], dataset['target'].shape[0]), 
                      dtype=np.float32)
  adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
  adj = normalise(adj + sp.eye(adj.shape[0]))
  adj = sparse_mx_to_sparse_tensor(adj)
  return adj
