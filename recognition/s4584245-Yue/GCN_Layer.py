import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as fun


class GCNLayer(nn.Module):
    """
    GCN layer class.
    """
    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        support = torch.mm(input_feature, self.weight.to(device))
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias.to(device)
        return output

class GCN(nn.Module):
    def __init__(self, input_dim=128):
        super(GCN, self).__init__()
        self.gcn1 = GCNLayer(input_dim, 16)
        self.gcn2 = GCNLayer(16, 4)

    def forward(self, adjacency, feature):
        h = fun.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits
