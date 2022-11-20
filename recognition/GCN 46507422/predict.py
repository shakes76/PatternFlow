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
from train import plot_tsne
from dataset import preprocess

MODEL_PATH = "./best_model.pt"
N_HID = 20
N_OUTPUT = 4

# load in adjacency matrix, features, labels and indices
adj_mtrx, node_features, labels, train_ids, val_ids, test_ids = preprocess()

# sample a node & subset of adj matrx
sample_adj = adj_mtrx[0:10, 0:10]
sample_features = node_features[0:10, :]

# construct model & evaluate
model = Net(node_features.shape[1], N_HID, N_OUTPUT)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# add node embeddings and evaluate:
node_embeddings = model((sample_features), (sample_adj))
test_gcn(node_embeddings, test=False)
# visualise node embeddings of sample using tSNE
plot_tsne(node_embeddings)

# visualise node embeddings of entire dataset using tSNE
node_embeddings_total = model((node_features), (adj_mtrx))
plot_tsne(node_embeddings_total, labels)
