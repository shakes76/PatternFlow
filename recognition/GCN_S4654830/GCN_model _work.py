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

# use torch method set mask train, mask test, mask validation, and adjacency matrix
tensor_x = features.to(device)
tensor_y = labels.to(device)
tensor_train_mask = torch.from_numpy(train_mask).to(device)
tensor_val_mask = torch.from_numpy(val_mask).to(device)
tensor_test_mask = torch.from_numpy(test_mask).to(device)
indices = torch.from_numpy(np.asarray([adjacency.row, adjacency.col]).astype('int64')).long()
values = torch.from_numpy(adjacency.data.astype(np.float32))
tensor_adjacency = torch.sparse.FloatTensor(indices, values, (22470, 22470)).to(device)

# set test logic fuction use the capsys as the argument which can calculate the accuracy and predict result.
def test(capsys):
    model.eval()
    with torch.no_grad():
        logits = model(tensor_adjacency, tensor_x)
        test_mask_logits = logits[capsys]


        predict_y = test_mask_logits.max(1)[1]
        accuarcy = torch.eq(predict_y, tensor_y[capsys]).float().mean()
    return accuarcy, test_mask_logits.cpu().numpy(), tensor_y[capsys].cpu().numpy()


