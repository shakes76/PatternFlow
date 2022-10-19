''' Contains source code of model components- each component is implemented as a class or function'''

# import networkx as nx
import numpy as np
# import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

# GNN LAYER- helper codes
class GNNLayer(Module):
  def __init__(self, in_features, out_features):
    super().__init()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(torch.FloatTensor(in_features, out_features))
    torch.nn.init.xavier_uniform_(self.weight)
  
  def forward(self, features, adj, active=True):
    # multiply features and weights
    support = torch.mm(features, self.weight)
    # perform matrix multiplication on sparse adjacency matrix and weighted features
    output = torch.spmm(adj, support)
    if active:
      # apply activation function over the top
      output = F.relu(output)
    return output

class Net(torch.nn.Module):
  # change this
  def __init__(self, features, hidden_layers, outputs):
    super(Net, self).__init__()
    self.first_conv = GNNLayer(features, hidden_layers)
    self.second_conv = GNNLayer(hidden_layers, outputs)
  
  def forward(self, input, adj):
    input = F.relu(self.first_conv(input, adj))
    # add in extra dropout
    input = F.dropout(input, 0.2, training=self.training)
    input = self.second_conv(input, adj)
    # add in extra drop out 
    input = F.dropout(input, 0.2, training=self.training)
    return F.softmax(input)
