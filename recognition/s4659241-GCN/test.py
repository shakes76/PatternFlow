    """
    code for testing the algorithm
    by: Kexin Peng, 4659241
    """
import numpy as np

facebook = np.load('facebook.npz')
edges = facebook['edges']
features = facebook['features']
target = facebook['target']

num_classes = len(np.unique(target))
num_nodes = features.shape[0]
num_features = features.shape[1]
num_edges = edges.shape[0]