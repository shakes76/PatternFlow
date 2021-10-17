import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class Facebook_Node_Classifier():
    def __init__(self, facebook_file: str):
        
        self.data_process(facebook_file)


    def data_process(self, facebook_file: str):
        data = np.load(facebook_file)

        edges = data['edges']
        features = data['features']
        target = data['target']

        self.n_edges = edges.shape[0]
        self.n_features = features.shape[1]
        self.n_target = target.shape[0]
        self.n_class = len(np.unique(target))

        self.node_features = torch.FloatTensor(features)
        self.target = torch.LongTensor(target)
        self.edges = edges
        
