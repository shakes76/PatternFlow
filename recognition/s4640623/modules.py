import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

"""Create a graph convolution layer (GCL) to be added in the GCN"""
class GraphConvolutionLayer(Module):

  def __init__(self, input_feat, output_feat, bias=True):
    super(GraphConvolutionLayer, self).__init__()
    self._input_feat = input_feat
    self._output_feat = output_feat
    self._weight = Parameter(torch.FloatTensor(self._input_feat, self._output_feat))
    self._bias = Parameter(torch.FloatTensor(self._output_feat))
    self.reset_parameters()

  """Sets weight and bias to be randomaly selected data points from a 
  uniform distribution"""
  def reset_parameters(self):
    stdv = 1. / math.sqrt(self._weight.size(1))
    self._weight.data.uniform_(-stdv, stdv)
    self._bias.data.uniform_(-stdv, stdv)

  """Calculates output tensor using GCL equation"""
  def forward(self, input, adj):
    support = torch.mm(input, self._weight)
    output = torch.spmm(adj, support)
    return output + self._bias

"""Create the Graph Convolution Network"""
class GCN(nn.Module):

  def __init__(self, input_feat, output_feat, nclass, dropout = 0.5):
    super(GCN, self).__init__()
    self._GraphConv1 = GraphConvolutionLayer(input_feat, output_feat)
    self._GraphConv2 = GraphConvolutionLayer(output_feat, nclass)
    self._dropout = dropout

  """Calculates output tensor by running data through GCN layers"""
  def forward(self, x, adj):
    # Relu activation
    x = func.relu(self._GraphConv1(x, adj))
    # Prevent co-adaption of feature detectors
    x = func.dropout(x, self._dropout, training=self.training)
    x = self._GraphConv2(x, adj)
    return func.log_softmax(x, dim=1)
