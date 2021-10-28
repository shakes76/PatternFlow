import torch.nn as nn
from GCN_Layer import *
import torch.nn.functional as fun

# GCN model import gcn layer class
class GCN(nn.Module):
    def __init__(self, input_dim=128):
        super(GCN, self).__init__()
        # set first gcn layer for gcn model
        self.gcn1 = GCN_Layer(input_dim, 16)
        # set second gcn layer for gcn model
        self.gcn2 = GCN_Layer(16, 4)

