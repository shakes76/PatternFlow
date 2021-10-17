import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Facebook_Node_Classifier():
    def __init__(self, facebook_file: str):
        
        self.data_process(facebook_file)
        self.create_adj()
        
    def data_process(self, facebook_file: str):
        data = np.load(facebook_file)

        edges = data['edges']
        features = data['features']
        target = data['target']

        self.n_edges = edges.shape[0]
        self.n_features = features.shape[1]
        self.n_target = target.shape[0]
        self.n_class = len(np.unique(target))

        self.node_features = torch.FloatTensor(features)
        self.target = torch.LongTensor(target)
        self.edges = edges
        
    def create_adj(self):
        #create an iniitial adj matrix for sparse matrix
        adj = sp.coo_matrix((np.ones(self.n_edges), (self.edges[:, 0], self.edges[:, 1])))
        
        #make sure all element is 1 or 0
        adj_t = torch.Tensor(adj.toarray())
        adj_t[adj_t > 0] = 1

        #check adj is semetric or not
        assert sum(sum(adj_t != adj_t.T)) == 0, 'Adjacency matrix is not symetric'

        self.adj = adj_t

if __name__ == "__main__":
    
    facebook_path = 'facebook.npz'

    classifer = Facebook_Node_Classifier(facebook_file=facebook_path)

    print(torch.unique(classifer.adj.unique()))

    
