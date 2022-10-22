import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
 

    def __init__(self, input_features, output_features, use_bias=True):
        super(GraphConvolution, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # weight in the layer
        self.use_bias=use_bias
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        # bias in the layer
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


    def forward(self, adj_matrix, in_feature):
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
    def __init__(self, in_feature=128):
        super(GCN, self).__init__()

        # first GraphConvolution layer
        self.layer1 = GraphConvolution(in_feature,32) 
        # second GraphConvolution layer
        self.layer2 = GraphConvolution(32, 8)  
        

    def forward(self, adjacency, feature):
        h = F.relu(self.gcn1(adjacency, feature))
        logits=self.gcn2(adjacency, h)
        return logits
