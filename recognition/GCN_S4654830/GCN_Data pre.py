import numpy as np
import scipy.sparse as sp
import torch
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

def load_data():
    """
    this function is used to load data
    :return: return the node and features and target from load data
    """
    dim_data = np.load("facebook.npz")
    n = len(dim_data["target"])
    x = np.zeros((n, n), dtype=np.float32)
    for i in dim_data["edges"]:
        x[i[0]][i[1]] = 1
    return sp.csr_matrix(x), sp.csr_matrix(dim_data["features"], dtype=np.float32), dim_data["target"]
                            
# normalize the data to 1.

def standard_adj(adjacency):
    """
    This function is used the
    :param adjacency: the adjacency is change the data into adjacency metrix
    :return: return metrix data
    """
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()

def standard_features(features):
    """
    This function us used to standardization the features
    :param features: select features from data
    :return: return the features details and number of features
    """
    return features / features.sum(1)

# review the data and set the variables
adjacency, features, labels = load_data()
# Use one hot encoder encode the metrix data
Onehot_encode = LabelBinarizer()
labels = Onehot_encode.fit_transform(labels)

# use the normalize function to normalize the data defineded
adjacency = standard_adj(adjacency)
features = standard_features(features)
features = torch.FloatTensor(np.array(features))
labels = torch.LongTensor(np.where(labels)[1])

# set the different useing data to make import from other method.
num_nodes = features.shape[0]
train_mask = np.zeros(num_nodes, dtype=bool)
val_mask = np.zeros(num_nodes, dtype=bool)
test_mask = np.zeros(num_nodes, dtype=bool)

# select the range of train and validation and test data
train_mask[:140] = True
val_mask[200:500] = True
test_mask[500:1500] = True
