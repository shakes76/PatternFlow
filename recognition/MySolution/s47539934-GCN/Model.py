"""
Author: Arsh Upadhyaya, 47539934
Code for 2 layer GCN model
"""
import math
import torch.nn.init as init
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
from random import sample
import matplotlib.pyplot as plt


class GraphConvolution(Module):
    '''
    Starting of graph convolutional layer.
    Parameters:
    input_features: dimensions of input layer
    output_features: dimenstions of output layer
    use_bias: optional but good practice
    '''
    
    def __init__(self, in_features, out_features, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias=use_bias
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if self.use_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
     #initialize parameters 
    def reset_parameters(self):
        self.weight = nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

'''
parameters:
in_feature: an n-dimenstional vector
adj_matrix: an adjacency matrix in tensor format
'''
    def forward(self, input, adj):
     
        support = torch.mm(input, self.weight) 
        output = torch.sparse.mm(adj, support)
       
        return output


class GCN(nn.Module):
 '''
 A model that contains 2 layers of GCN , by creating 2 instances from GraphConvolution function
 parameters:
 in_feature:n dimensional vector, which is input
 out_class: n dimensional vector, final output
 in this case model goes 128->32->4
 since in_feature=128(known from dataset)
 out_class=4(since finally 4 classes)
 
 '''
    def __init__(self, in_feature,  out_class, dropout):
        super(GCN, self).__init__()

        self.gcn_conv_1 = GraphConvolution(in_feature, 32)#32 is like the hidden layer for the overall model
        self.gcn_conv_2 = GraphConvolution(32, out_class)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gcn_conv_1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gcn_conv_2(x, adj)

        return F.log_softmax(x, dim=1)
