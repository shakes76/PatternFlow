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

# set the training method logic fuction to calculate the loss history list and validation accuracy history
def train():
    # set list to storage the calculate accuracy,the calculate method will use the for loop
    loss_history = []
    val_acc_history = []
    model.train()
    train_y = tensor_y[tensor_train_mask]
    # epoch value already set as 200
    for epoch in range(epochs):
        logits = model(tensor_adjacency, tensor_x)
        train_mask_logits = logits[tensor_train_mask]
        # loss calculate method
        loss = criterion(train_mask_logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc, _, _ = test(tensor_train_mask)
        val_acc, _, _ = test(tensor_val_mask)
        loss_history.append(loss.item())
        val_acc_history.append(val_acc.item())
        # out put the accuracy in loss and train and validation data
        print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
            epoch, loss.item(), train_acc.item(), val_acc.item()))

    return loss_history, val_acc_history
