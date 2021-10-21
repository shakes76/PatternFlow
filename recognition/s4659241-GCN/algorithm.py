"""
code for the algorithm
by: Kexin Peng, 4659241
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(Module):
    """
    Graph Convolution layer
    references: 
    https://arxiv.org/abs/1609.02907
    https://github.com/tkipf/pygcn/tree/1600b5b748b3976413d1e307540ccc62605b4d6d

    """

    def __init__(self, input_features, output_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # weight in the layer
        self.weight = Parameter(torch.FloatTensor(input_features, output_features))
        # bias in the layer
        if bias:
            self.bias = Parameter(torch.FloatTensor(output_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    # initialize parameters using kaiming-uniform
    def reset_parameters(self):
        self.weight = nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            p = 1./math.sqrt(len(self.bias))
            self.bias.data.uniform_(-p, p)


    def forward(self, in_feature, adj_matrix):
        # input * weight
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

# 2 layer GCN
class GCN(nn.Module):
    def __init__(self, n_feature, n_hidden, n_class, dropout):
        super(GCN, self).__init__()
        
        # first GraphConvolution layer
        self.layer1 = GraphConvolution(n_feature, n_hidden) 
        # second GraphConvolution layer
        self.layer2 = GraphConvolution(n_hidden, n_class)  
        self.dropout = dropout

    def forward(self, x, adj_matrix):
        x = F.relu(self.layer1(x, adj_matrix))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.layer2(x, adj_matrix)
        return F.log_softmax(x, dim=1)

class GCN_3l(nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_class, dropout):
        super(GCN_3l, self).__init__()
        
        # first GraphConvolution layer
        self.layer1 = GraphConvolution(n_feature, n_hidden1) 
        # second GraphConvolution layer
        self.layer2 = GraphConvolution(n_hidden1, n_hidden2) 
        # third GraphConvolution layer
        self.layer3 = GraphConvolution(n_hidden2, n_class)  
        self.dropout = dropout

    def forward(self, x, adj_matrix):
        x = F.relu(self.layer1(x, adj_matrix))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.layer2(x, adj_matrix))
        x = self.layer3(x, adj_matrix)
        return F.log_softmax(x, dim=1)
