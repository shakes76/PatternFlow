import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    GCN Layer
    """

    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = Parameter(torch.FloatTensor(out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adjacency_matrix):
        x = torch.mm(x, self.weight)
        x = torch.spmm(adjacency_matrix, x)
        return x + self.bias


class GCN(nn.Module):
    """
    Model
    """

    def __init__(self, input_size, hidden_size, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.gconv1 = GraphConvolution(input_size, hidden_size)
        self.gconv2 = GraphConvolution(hidden_size, num_classes)
        self.dropout = dropout

    def forward(self, x, adjacency_matrix):
        x = self.gconv1(x, adjacency_matrix)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gconv2(x, adjacency_matrix)
        return F.log_softmax(x, dim=1)
