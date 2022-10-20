'''contains the source code for training, validating, testing and saving model'''
import modules as m
import datasets as ds
import torch
import torch.nn as nn


# Load data
adj_mtrx, node_features, labels, train_ids, val_ids, test_ids = ds.preprocess()
node_features = node_features.double()
adj_mtrx = adj_mtrx.double()
val_loss = []
train_loss = []
val_acc = []
train_acc = []

gcn = m.Net(node_features.shape[1], 65, 4)
# GCN(nfeat=features.shape[1], nhid=65, nclass=4, dropout=0.2)
optimiser = torch.optim.Adam(gcn.parameters())

# GCN = Net(features, 2, 4)
def pred_accuracy(pred, labels):
    acc = (pred.max(1)[1] == labels).sum() / len(labels)
    return acc

def train(epoch):
    gcn.train()
    # reset grads
    gcn.zero_grad(set_to_none=True)
    output = gcn(node_features, adj_mtrx)
    print("output", output.shape)
    loss_fn = nn.CrossEntropyLoss()
    training_loss = loss_fn(output[train_ids], labels[train_ids])
    training_accuracy = pred_accuracy(output[train_ids], labels[train_ids])
    training_loss.backward()
    optimiser.step()

    gcn.eval()
    output = gcn(node_features, adj_mtrx)

    loss_val = loss_fn(output[val_ids], labels[val_ids])
    
    acc_val = pred_accuracy(output[val_ids], labels[val_ids])
    print("val", output[val_ids])
    print("train", output[train_ids])
    print(f'Epoch: {epoch + 1}',
          f'Training Loss: {training_loss.item()}',
          f'Training Accuracy: {training_accuracy.item()}',
          f'Validation Loss: {loss_val.item()}',
          f'Validation Accuracy: {acc_val.item()}')

val_loss = []
train_loss = []
val_acc = []
train_acc = []
# Load data
train_adj, val_adj, test_adj, features, labels, idx_train, idx_val, idx_test = ds.preprocess()

features = features.double()
print(labels)
labels = labels.long()
train_adj = train_adj.double()
val_adj = val_adj.double()
test_adj = test_adj.double()
# print(adj[idx_train][idx_train].shape)
# # Model and optimizer
# print(adj[[idx_train, idx_train]].size())
# print(adj.size())

# print(labels.max().item())
gcn = m.Net(features.shape[1], 65, 4)
# GCN(nfeat=features.shape[1], nhid=65, nclass=4, dropout=0.2)
optimiser = optim.Adam(gcn.parameters())

# GCN = Net(features, 2, 4)

def train_gcn(epoch):
    gcn.train()
    # reset grads
    gcn.zero_grad(set_to_none=True)
    output = gcn(features[idx_train], train_adj)
    # print("index", idx_train)
    print("output", output.shape)
    # print("output train", output[idx_train].shape)
    # print(output[idx_train])
    # print("labels", labels.shape)
    # print(labels[idx_train])
    loss_fn = nn.CrossEntropyLoss()
    training_loss = loss_fn(output, labels[idx_train])
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # acc_train = (labels[idx_train] == output[idx_train].max(1).item() / labels[idx_train].size(0))
    training_acc = pred_accuracy(output, labels[idx_train])
    training_loss.backward()
    optimiser.step()

    gcn.eval()
    output = gcn(features[idx_val], val_adj)

    val_loss = loss_fn(output, labels[idx_val])
    
    acc_val = pred_accuracy(output, labels[idx_val])
    val_loss.append(val_loss.item())
    train_loss.append(training_loss.item())
    val_acc.append(acc_val)
    train_acc.append(training_acc)

    # print("val", output[idx_val])
    # print("train", output[idx_train])
    print(f'Epoch: {epoch + 1}',
          f'loss_train: {train_loss.item()}',
          f'acc_train: {train_acc.item()}',
          f'loss_val: {val_acc.item()}',
          f'acc_val: {acc_val.item()}')


