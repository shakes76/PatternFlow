'''contains data loader for loading and preprocessing data'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
# import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy.linalg as linalg
from torch.utils.data import DataLoader
import torch 
# from torch_geometric.utils.convert import from_networkx
import scipy.sparse as sp
from sklearn.preprocessing import LabelBinarizer

data = np.load('./facebook.npz')
lst = data.files
print(lst)
data['edges'].shape

def rownormalise(inp):
  """
  Used to help normalise the rows of your adjacency and feature matrix 
  https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py 
  
  """
   
  summed_rows = np.array(inp.sum(1))
  # print("rowsum", rowsum)
  print("before inv")
  row_inverse = (np.reciprocal(summed_rows)).flatten()
  #np.power(rowsum, -1).flatten()
  #
  # print("inverse", r_inv)
  print("after inv")

  row_inverse[np.isinf(row_inverse)] = 0.
  inverted_matrix = sp.diags(row_inverse)
  
  # return normalised row 
  inp = inverted_matrix.dot(inp)
  # print(mx)
  return inp
    
def preprocess():
  data = (np.load('./facebook.npz'))
  # print(data)
  lst = data.files
  print(lst)

  # print((data['features']).shape)
  # train  test split 

  #Initialize the graph
  G = nx.Graph(name='G')
  edges = (data['edges']).tolist()
  features_list = (data['features']).tolist()
  feature_names = (np.arange(0, 128)).tolist()

  from sklearn.utils import shuffle
  #shuffle targets and features
  # print(data['features'].tolist())
  features, target = ((data['features']), data['target'])

  # get a feature dictionary 
  features_df = pd.DataFrame(features, columns=np.arange(0, 128))
  feature_dict = features_df.to_dict(orient='records')
  features_ = [tuple([i, f]) for i, f in enumerate(feature_dict)]
  # print((features_df))

  # print("Total number of self-loops: ", len(list(nx.selfloop_edges(G))))
  print("before nodes!!")
  G.add_nodes_from(features_, name=np.arange(1, G.number_of_nodes()))

  # print(edges)
  G.add_edges_from(edges)
  G_ = G.copy()

  # print(edges[0])
  edges = [tuple(e) for e in edges]
  self_loops = [tuple((n, n)) for n in range(G.number_of_nodes())]
  print("before edges!!")
  G_.add_edges_from(self_loops)

  A = nx.to_numpy_array(G)
  # print(A)
  X = features_list

  A_copy = A.copy()

  # print(A_copy)
  # print("adj", A_copy[np.nonzero(A_copy)])

  np.fill_diagonal(A, 1)
  # print(A)
  # print(A_copy)
  print("added loops!!")
  diags = np.diag(A) + np.diag(A_copy)
  # print(diags)
  np.fill_diagonal(A_copy, diags)

  
  # print(A_copy)
  print("before dot!")
  # AX = np.dot(A_copy,X)
 
  # D_inverse = linalg.fractional_matrix_power(linalg.inv(deg_mat), -0.5)
  # D_inv_AX = np.dot(D_inverse, AX)

  # normalised_adjacency = D_inv_AX
  # print("here!")
  adj = rownormalise(A_copy)#torch.from_numpy(rownormalise(A_copy))
  print("sum", np.array(features.sum(1)))
  features = rownormalise(features)
  # print(int(0.5*len(features)))
  idx_train = np.arange(int(0.6*len(features)))
  # print("train", idx_train)
  idx_val = np.arange(int(0.6*len(features)), int(0.6*len(features)) + int(0.2*len(features)))
  # print("val", idx_val)
  idx_test = np.arange(int(0.6*len(features)) + int(0.2*len(features)), len(features))
  
  # print("test", idx_test)
  idx_train = torch.LongTensor(idx_train)
  idx_val = torch.LongTensor(idx_val)
  idx_test = torch.LongTensor(idx_test)


 
  train_adj = torch.from_numpy(adj[idx_train,:][:, idx_train])
  val_adj = torch.from_numpy(adj[idx_val, :][:, idx_val])
  test_adj = torch.from_numpy(adj[idx_test, :][:, idx_test])

  encoder_target = LabelBinarizer()
  encoded = encoder_target.fit_transform(target)
  
  labels = torch.LongTensor(np.where(encoded)[1])
  # print(labels)
  features = torch.FloatTensor((features))
  # print("before adj!")
  
  return train_adj, val_adj, test_adj, features, labels, idx_train, idx_val, idx_test