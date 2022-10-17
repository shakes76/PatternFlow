'''contains the source code for training, validating, testing and saving model'''
import modules as m
import datasets as ds
import torch
import torch.nn as nn


# Load data
adj, features, labels, idx_train, idx_val, idx_test = ds.preprocess()
features = features.double()
adj = adj.double()
# Model and optimizer
print(idx_train)

print(labels.max().item())
gcn = m.Net(features.shape[1], 65, 4)
# GCN(nfeat=features.shape[1], nhid=65, nclass=4, dropout=0.2)
optimiser = torch.optim.Adam(gcn.parameters())

# GCN = Net(features, 2, 4)
def accuracy_function(output, labels):
    prediction = output.max(1)[1].type_as(labels)
    # intersection betwen predicted and actual test labels
    intersect = prediction.eq(labels).double()
    intersect = intersect.sum()
    return intersect / len(labels)

def train(epoch):
    gcn.train()
    # reset grads
    gcn.zero_grad(set_to_none=True)
    output = gcn(features, adj)
    print("output", output.shape)
    loss_fn = nn.CrossEntropyLoss()
    loss_train = loss_fn(output[idx_train], labels[idx_train])
    acc_train = accuracy_function(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimiser.step()

    gcn.eval()
    output = gcn(features, adj)

    loss_val = loss_fn(output[idx_val], labels[idx_val])
    
    acc_val = accuracy_function(output[idx_val], labels[idx_val])
    print("val", output[idx_val])
    print("train", output[idx_train])
    print(f'Epoch: {epoch + 1}',
          f'loss_train: {loss_train.item()}',
          f'acc_train: {acc_train.item()}',
          f'loss_val: {loss_val.item()}',
          f'acc_val: {acc_val.item()}')
