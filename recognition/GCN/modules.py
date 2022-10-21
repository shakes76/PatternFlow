import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

DROPOUT = 0.3
class GNNLayer(Module):
  def __init__(self, in_features, out_features):
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.weight = Parameter(torch.DoubleTensor(self.in_features, self.out_features))
    torch.nn.init.xavier_uniform_(self.weight)
  
  def forward(self, features, adj):
    features = features.double()
    # multiply features and weights
    support = torch.mm(features, self.weight)
    # perform matrix multiplication on sparse adjacency matrix and weighted features to average over neighbours
    output = torch.spmm(adj, support)
    # apply activation function over the top 
    output = F.relu(output)
    return output

class Net(torch.nn.Module):
  # construct a GCN with three hidden layers
  def __init__(self, features, hidden_layers, outputs):
    super(Net, self).__init__()
    self.outputs = outputs 
    self.features = features
    self.hidden_layers = hidden_layers
    self.first_conv = GNNLayer(features, hidden_layers)
    self.second_conv = GNNLayer(hidden_layers, hidden_layers)
    self.third_conv = GNNLayer(hidden_layers, outputs)
  
  def forward(self, input, adj):
    # activate first layer
    input = self.first_conv(input, adj)
    # add dropout to prevent overfitting
    input = F.dropout(input, DROPOUT, training=self.training)
    # apply convolution to second layer
    input = self.second_conv(input, adj)
     # add dropout to prevent overfitting
    input = F.dropout(input, DROPOUT, training=self.training)
    input = self.third_conv(input, adj)
    return F.softmax(input)