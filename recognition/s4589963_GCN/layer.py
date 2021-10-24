import torch

"""
Graph convolution network layer.
"""


class GCNLayer(torch.nn.Module):
    """
    initializer of graph convolution layer.

    Params:
    input_dim: input dimension of this layer.
    out_dim: output dimension of this layer.
    use_bias: if use bias(optional).
    """

    def __init__(self, input_dim, output_dim, use_bias=True):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias
        self.weight = torch.nn.Parameter(torch.Tensor(input_dim, output_dim))
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.Tensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    """
    initialise parameters.
    """

    def reset_parameters(self):
        # initialise weight.
        torch.nn.init.kaiming_uniform_(self.weight)
        if self.use_bias:
            torch.nn.init.zeros_(self.bias)

    """
    Define computation performed at every call.

    Params:
    adjacency: adjacency matrix.
    input_feature: features of every data in dataset.
    """

    def forward(self, adjacency, input_feature):
        device = "cpu"
        support = torch.mm(input_feature, self.weight.to(device))
        # adjacency is sparse matrix so it need torch.sparse.mm instead of torch.mm
        output = torch.sparse.mm(adjacency, support)
        if self.use_bias:
            output += self.bias.to(device)
        return output
