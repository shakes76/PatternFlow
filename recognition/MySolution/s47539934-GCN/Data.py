import numpy as np
from sklearn import preprocessing
import torch
import scipy.sparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import scipy.sparse as sp
import torch.optim as optim


def load_data(file_path):
  data=np.load(file_path)
  edges= data['edges']
  features=data['features']
  target=data['target']

def normalize_adj(matrix):
  matrix += sp.eye(matrix.shape[0])
  degree = np.array(matrix.sum(1))
  d_hat = sp.diags(np.power(degree, -0.5).flatten())
  return d_hat.dot(matrix).dot(d_hat).tocoo()

file_dir="/content/drive/MyDrive/facebook.npz"
load_data(file_dir)
features=preprocessing.normalize(features)
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(features.shape[0], features.shape[0]),
                        dtype=np.float32)

adj=normalize_adj(adj)
num_nodes = features.shape[0]
num_features = features.shape[1]
num_edges = edges.shape[0]
train_set = torch.LongTensor(range(int(num_nodes*0.2)))
val_set = torch.LongTensor(range(int(num_nodes*0.2),int(num_nodes*0.4)))
test_set = torch.LongTensor(range(int(num_nodes*0.4),num_nodes))