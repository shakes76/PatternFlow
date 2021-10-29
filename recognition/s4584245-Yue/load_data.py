#Import the necessay packages.
import numpy as np
import scipy.sparse as sp
import torch

fb = np.load('facebook.npz')
#The details of the data-set
target = fb['target']
print(target.shape)
edges = fb['edges']
print(edges.shape)
feature = fb['features']
print(feature.shape)

def load_data():
    """
    load the data
    """
    fb = np.load("facebook.npz")
    n = len(target)
    x = np.zeros((n, n), dtype=np.float32)
    for i in edges:
        x[i[0]][i[1]] = 1
    return sp.csr_matrix(x), sp.csr_matrix(feature, dtype=np.float32), target

load_data()

