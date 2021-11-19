#Import the GCN model.
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as f


class GCNLayer(nn.Module):
    def __init__(self, in_feature, out_feature, bias=True):
        """
        Parameters:
        in_feature: Dimension of nodes input feature.
        out_feature: Dimension of nodes input feature.
        bias: bool. Use bias or not.
        """
        super(GCNLayer, self).__init__()
        self.input_dim = in_feature
        self.output_dim = out_feature
        self.tf_bias = bias
        self.weight = nn.Parameter(torch.Tensor(in_feature, out_feature))

        if self.tf_bias:
            self.bias = nn.Parameter(torch.Tensor(out_feature))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.tf_bias:
            init.zeros_(self.bias)

    def forward(self, adjacency, input_feature):
        """
        The adjacency matrix is a sparse matrix,
        so sparse matrix multiplication is used in the calculation

        Parameters:
        adjacency: torch.sparse.FloatTensor
        input_feature: torch.Tensor
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        support = torch.mm(input_feature, self.weight.to(device))
        output = torch.sparse.mm(adjacency, support)
        if self.tf_bias:
            output += self.bias.to(device)
        return output

class GCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907(Reference)
    """
    def __init__(self, in_feature = 128):
        """
        Define a model with two layers of GraphConvolution.
        """
        super(GCN, self).__init__()
        #input_feature: 128
        self.gcn1 = GCNLayer(in_feature, 16)
        #output: 4 categories
        self.gcn2 = GCNLayer(16, 4)

    def forward(self, adjacency, feature):
        h = f.relu(self.gcn1(adjacency, feature))
        logits = self.gcn2(adjacency, h)
        return logits
