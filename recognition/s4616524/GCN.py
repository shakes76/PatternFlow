import math
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

class GCNLayer(Module):
    def __init__(self, n_in_features, n_out_features):
        super(GCNLayer, self).__init__()
        self.n_in_features = n_in_features
        self.n_out_features = n_out_features
        self.weights = Parameter(torch.FloatTensor(n_in_features, n_out_features))

    def forward(self, input, adj):
        linear = torch.mm(input, self.weights) #WX
        output =  torch.mm(adj, linear) #D^-1*A*WX

        return output

class GCNModel(nn.Module):
    def __init__(self, n_class, n_in_features):
        super(GCNModel, self).__init__()

        self.gcn1 = GCNLayer(n_in_features, 64)
        self.gcn2 = GCNLayer(64, 32)
        self.ffn = nn.Linear(32, n_class)

    def forward(self, input:torch.FloatTensor, adj:torch.FloatTensor):
        x1 = F.relu(self.gcn1(input, adj))
        x2 = F.relu(self.gcn2(x1, adj))
        x3 = self.ffn(x2)
        
        result = F.softmax(x3, dim=1)

        return result
        

class Facebook_Node_Classifier():
    def __init__(self, facebook_file: str):
        
        self.data_process(facebook_file)
        self.create_adj()
        self.model = GCNModel(self.n_class, self.n_features)
        
    def data_process(self, facebook_file: str):
        data = np.load(facebook_file)

        edges = data['edges']
        features = data['features']
        target = data['target']

        self.n_edges = edges.shape[0]
        self.n_features = features.shape[1]
        self.n_node = target.shape[0]
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

        #normalise 
        rowsum = np.array(adj_t.sum(1))
        inv = np.ma.power(rowsum, -1) #creaet D^-1
        inv[inv == np.inf] = 0.#if 0 is inv 
        D_inv = sp.diags(inv)
        adj_m = D_inv.dot(adj_t) #D^-1*A

        self.adj = torch.FloatTensor(adj_m)

    def get_acc(self, output:torch.FloatTensor):
        prediction = output.argmax(1)
        correct = output == self.target
        
        return sum(correct)/len(self.n_node)
        

    def train_modle(self, n_epoch=30, lr=0.01):
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(n_epoch):
            self.model.train()
            optimizer.zero_grad()
            
            output = self.model(self.node_features, self.adj)
            prediction_prob = output[self.target]
            loss = -torch.mean(torch.log(prediction_prob))

            loss.backward()
            optimizer.step()

            print("Epoch: " + str(epoch) + " Loss: " + str(loss))



if __name__ == "__main__":
    
    facebook_path = 'facebook.npz'

    classifer = Facebook_Node_Classifier(facebook_file=facebook_path)

    #classifer.train_modle()

    
