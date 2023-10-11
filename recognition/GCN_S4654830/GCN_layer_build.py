import torch
import torch.nn as nn
from torch.nn import init

class GCN_Layer_build(nn.Module):
    """
    This class is used to build the GCN layer to structure the GCN model
    """
    def __init__(self, input_dim, output_dim, use_bias=True):
        # set all the variables
        super(GCN_Layer_build, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.reset_parameters()
        self.weight = nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
    # use this function to reset the parameters by weight and self.bias
    def reset_parameters(self):
        init.kaiming_uniform_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)
    # this fuction is return the output based on forward
    def forward(self, adjacency, input_feature):
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        support = torch.mm(input_feature, self.weight.to(device))
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias.to(device)
        return output