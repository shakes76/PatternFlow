import modules as m
import train as t
# import networkx as nx
import numpy as np
# import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import scipy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from sklearn.manifold import TSNE
import torch
from modules import Net, GNNLayer
from train import test_gcn
from dataset import preprocess

PATH = "/Users/maryamkhan/Documents/UNI/2022/SEM2/COMP3710/PatternFlow/recognition/GCN/best_model.pt"
N_EPOCHS = 5
NHID = 4
NOUTPUT = 4

adj_mtrx, node_features, labels, train_ids, val_ids, test_ids = preprocess()
sample_adj = adj_mtrx[0]
sample_features = node_features[0]
print(sample_features)

# device_model = torch.device("cuda")
model = m.Net(node_features.shape[1], NHID, NOUTPUT)
model.load_state_dict(torch.load(PATH))
# model.to(device_model)

# print("saved", model)
model.eval()
gcn = model(torch.FloatTensor(np.asmatrix(sample_features)), torch.from_numpy(np.asmatrix(sample_adj)))
test_gcn(gcn)
