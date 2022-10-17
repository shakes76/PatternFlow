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

  print((data['features']).shape)
  # train  test split 

  #Initialize the graph
  G = nx.Graph(name='G')
  edges = (data['edges']).tolist()
  features_list = (data['features']).tolist()
  feature_names = (np.arange(0, 128)).tolist()
  # print(data['features'].tolist())

  # get a feature dictionary 
  features_df = pd.DataFrame(data['features'], columns=np.arange(0, 128))
  feature_dict = features_df.to_dict(orient='records')
  features = [tuple([i, f]) for i, f in enumerate(feature_dict)]
  print(len(features_df))

  # print("Total number of self-loops: ", len(list(nx.selfloop_edges(G))))
  print("before nodes!!")
  G.add_nodes_from(features, name=np.arange(1, G.number_of_nodes()))

  # print(edges)
  G.add_edges_from(edges)
  G_ = G.copy()

  # print(edges[0])
  edges = [tuple(e) for e in edges]
  self_loops = [tuple((n, n)) for n in range(G.number_of_nodes())]
  print("before edges!!")
  G_.add_edges_from(self_loops)

  A = nx.to_numpy_array(G)
  X = features_list

  A_copy = A

  np.fill_diagonal(A, 1)
  print("added loops!!")
  diags = np.diag(A) + np.diag(A_copy)
  np.fill_diagonal(A_copy, diags)

  # print(A_copy.shape)
  print("before dot!")
  AX = np.dot(A_copy,X)
  # deg_mat = G_.degree()
  # print(G.degree[0])
  # for (n,deg) in list(deg_mat):
  #   print(n, deg)
  # deg_mat = np.diag([deg for (n,deg) in list(deg_mat)])
  # print(type(deg_mat))
  # 
  # deg_mat = torch.from_numpy(deg_mat)
  # deg_mat = deg_mat.float()
  # print(deg_mat)
  # print(normalize(deg_mat))
  # D_inverse = linalg.inv(deg_mat)
  # D_inverse = torch.linalg.inv(deg_mat)


  # D_inverse = linalg.fractional_matrix_power(linalg.inv(deg_mat), -0.5)
  # D_inv_AX = np.dot(D_inverse, AX)



  # print(np.array(adj.sum(1)))
  # features = torch.FloatTensor(np.array(features).todense())
  labels = torch.LongTensor(data['target'])
  # D_inverse = linalg.fractional_matrix_power(linalg.inv(deg_mat), -0.5)
  # D_inv_AX = np.dot(D_inverse, AX)

  # normalised_adjacency = D_inv_AX
  print("here!")
    

  features = torch.FloatTensor(rownormalise(data['features']))
  print("before adj!")
  adj = torch.from_numpy(rownormalise(A_copy))
  
  idx_train = np.arange(int(0.5*len(data)))
  print("train", idx_train)

  idx_val = np.arange(int(0.5*len(data)), int(0.5*len(data)) + int(0.25*len(data)))
  print("val", idx_val)
  idx_test = np.arange(int(0.5*len(data)) + int(0.25*len(data)), len(data))
  
  # print("test", idx_test)
  idx_train = torch.LongTensor(idx_train)
  idx_val = torch.LongTensor(idx_val)
  idx_test = torch.LongTensor(idx_test)

  # print(idx_train)
  # print(idx_test)
  # print("featured", type(features[0]))
  # print("edges", type(edges[0][0]))
  # print("self loops", type(self_loops[0][0]))
  # print(adj)
  # pyg_graph = from_networkx(G_)
  # degrees = np.matrix([val for (node, val) in G_.degree()])[:,0:100]
  # print(degrees.shape)
  # degrees = np.reshape(degrees, (10, 10))
  # print(degrees.shape)

  # print(nx.to_numpy_matrix(deg_mat))
  # print(deg_mat.shape)

  # print(np.linalg.inv)
  # deg_mat = tf.constant(deg_mat)
  # tf.cast(deg_mat, tf.float32)
  return adj, features, labels, idx_train, idx_val, idx_test

