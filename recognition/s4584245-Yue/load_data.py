#Import the necessay packages.
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelBinarizer

fb = np.load('facebook.npz')
#The details of the data-set
target = fb['target']
#print(target.shape)
edges = fb['edges']
#print(edges.shape)
feature = fb['features']
#print(feature.shape)

def load_data():
    """
    load the data
    """
    n = len(target)
    x = np.zeros((n, n), dtype=np.float32)
    for i in edges:
        x[i[0]][i[1]] = 1
    return sp.csr_matrix(x), sp.csr_matrix(feature, dtype=np.float32), target

def normalize_adj(adjacency):
    """
    Take the mean of the feature information of the neighbors.
    """
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()

def normalize_features(features):
    """
    Take the mean of the feature information of the neighbors.
    """
    return features / features.sum(1)

# Build the graph
adjacency, features, labels = load_data()
encode_onehot = LabelBinarizer()
labels = encode_onehot.fit_transform(labels)

adjacency = normalize_adj(adjacency)
features = normalize_features(features)
features = torch.FloatTensor(np.array(features))
labels = torch.LongTensor(np.where(labels)[1])

# Split the dataset.
num_nodes = features.shape[0]
train_mask = np.zeros(num_nodes, dtype=bool)
val_mask = np.zeros(num_nodes, dtype=bool)
test_mask = np.zeros(num_nodes, dtype=bool)
# train data
train_mask[:10000] = True
# valid data
val_mask[10000:12000] = True
# test data
test_mask[12000:15000] = True
