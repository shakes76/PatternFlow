"""
Author: Arsh Upadhyaya, 47539934
Code for 2 layer GCN model
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    '''
    Starting of graph convolutional layer.
    Parameters:
    input_features: dimensions of input layer
    output_features: dimenstions of output layer
    use_bias: optional but good practice
    '''
    def __init__(self, input_features, output_features, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.use_bias=use_bias
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    # initialize parameters using kaiming-uniform
    def reset_parameters(self):
        self.weight = nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

'''
parameters:
in_feature: an n-dimenstional vector
adj_matrix: an adjacency matrix in tensor format
'''
    def forward(self, adj_matrix, in_feature):
     
        support = torch.mm(in_feature, self.weight) 
        output = torch.sparse.mm(adj_matrix, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.input_features) + ' -> ' \
               + str(self.output_features) + ')'

class GCN(nn.Module):
 '''
 A model that contains 2 layers of GCN , by creating 2 instances from GraphConvolution function
 '''
    def __init__(self, in_feature=128):
        super(GCN, self).__init__()

        # first GraphConvolution layer taking input of 128 as that is format in facebook.npz
        self.layer1 = GraphConvolution(in_feature,32) 
        # second GraphConvolution layer having 32 as input, which is output from previous layer
        self.layer2 = GraphConvolution(32, 8)  
        

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits=self.gcn2(adjacency, h)
        return logits
