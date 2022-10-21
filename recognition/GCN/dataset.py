import numpy as np
import pandas as pd
import networkx as nx
import numpy as np
import pandas as pd
import torch 
import scipy.sparse as sp
from sklearn.preprocessing import LabelBinarizer

PATH = './facebook.npz'

def rownormalise(inp):
  """
  Used to  normalise rows of adjacency matrix
  https://github.com/tkipf/pygcn/blob/master/pygcn/utils.py 
  """
  # normalise rows
  summed_rows = np.array(inp.sum(1))
  # inverse of degree matrix 
  row_inverse = (np.reciprocal(summed_rows)).flatten()
  # turn any division by 0 into 0
  row_inverse[np.isinf(row_inverse)] = 0.
  # dot product of degree and adjacency
  inverted_matrix = sp.diags(row_inverse)

  # return normalised row 
  inp = inverted_matrix.dot(inp)
  return inp
    
def preprocess():
  data = np.load(PATH)
  lst = data.files
  features, target = (data['features']), (data['target'])

  #Initialize the graph
  G = nx.Graph(name='G')
  edges = (data['edges']).tolist()

  # get a feature dictionary 
  features_df = pd.DataFrame(features, columns=np.arange(0, features.shape[1]))

  # turn features into a dictionary with edges
  feature_dict = features_df.to_dict(orient='records')
  features_ = [tuple([i, f]) for i, f in enumerate(feature_dict)]

  # add nodes to features
  G.add_nodes_from(features_, name=np.arange(1, G.number_of_nodes()))

  # add edges to graph 
  G.add_edges_from(edges)
  G_ = G.copy()
  self_loops = [tuple((n, n)) for n in range(G.number_of_nodes())]
  G_.add_edges_from(self_loops)
  A = nx.to_numpy_array(G)

  A_copy = A.copy()
  # add diagonal of 1s to A
  np.fill_diagonal(A, 1)

  # add this to existing connections in diagonal 
  diags = np.diag(A) + np.diag(A_copy)
  # replace diagonal 
  np.fill_diagonal(A_copy, diags)

  # adjacency matrix = row normalised matrix with self loops 
  adj_mtrx = torch.from_numpy(rownormalise(A_copy)) 
  features = rownormalise(features)

  # reserve 60% for training,  20% for validation, and another 20% for testing
  train_end = int(0.6*len(features))
  val_end = int(0.6*len(features)) + int(0.2*len(features))
  test_end = len(features)

  idx_train = torch.LongTensor(np.arange(train_end))
  idx_val = torch.LongTensor(np.arange(train_end, val_end))
  idx_test = torch.LongTensor(np.arange(val_end, test_end))

  encoder_target = LabelBinarizer()
  encoded = encoder_target.fit_transform(target)
  labels = torch.LongTensor(np.where(encoded)[1])

  features = torch.DoubleTensor((features))

  return adj_mtrx, features, labels, idx_train, idx_val, idx_test



