from Data_prepare import *
from GCN_Layer import *
from  GCN_model_define import *
import numpy as np
import scipy.sparse as sp
import torch

import torch.nn as nn
import torch.nn.functional as fun
import torch.nn.init as init
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelBinarizer

# Hyperparameter settings
learning_rate = 0.1
weight_decay = 5e-4
epochs = 200

device = "cpu"
model = GCN().to(device)
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)



