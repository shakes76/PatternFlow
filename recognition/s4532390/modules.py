import torch
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter

class GCNLayer(Module):
    """
    A class for a single layer Graph convolutional network
    """
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    """
    Inputs:
        features - an n-dimensional vector (normalised)
        adjacency_matrix - an adjacency matrix in coo tensor format
        active - can deactivate relu activation function
    """
    def forward(self, features, adjacency_matrix, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adjacency_matrix, support)
        if active:
            output = F.relu(output)
        return output


class GCN(torch.nn.Module):
    """
    The graph convolutional model used for training against the facebook page data
    """
    def __init__(self, in_features, out_classes):
        super().__init__()
        # Uses two GCN layers
        self.conv1 = GCNLayer(in_features, 64)
        self.conv2 = GCNLayer(64, 32)
        self.conv3 = GCNLayer(32, out_classes)

    def forward(self, x, adjacency_matrix):
        """
        Structure:
            Convolutional layer (? -> 64)
            Relu + Dropout Layer
            Convolutional layer (64 -> 32)
            Relu + Dropout Layer
            Convolutional layer (32 -> 4)
            Logarithmic softmax
        """
        x = self.conv1(x, adjacency_matrix)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adjacency_matrix)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, adjacency_matrix)

        return F.log_softmax(x, dim=1)
