from layer import GCNLayer
import torch

"""
Graph convolution network.
Consist of two gcn layers.
"""


class GCN(torch.nn.Module):
    """
    initialise the gcn model.
    Param:
    input_dim: the input dimension of the features.(128 dimensions of facebook dataset.)
    """

    def __init__(self, input_dim=128):
        super(GCN, self).__init__()
        self.gcn_layer1 = GCNLayer(input_dim, 16)

        # 4 classes in this dataset.
        self.gcn_layer2 = GCNLayer(16, 4)

    """
    Define computation performed at every call.

    Params:
    adjacency: adjacency matrix.
    feature: features of every data in dataset.
    """

    def forward(self, adjacency, feature):
        hidden = torch.nn.functional.relu(self.gcn_layer1(adjacency, feature))
        output = self.gcn_layer2(adjacency, hidden)
        return output
