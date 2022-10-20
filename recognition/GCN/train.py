'''contains the source code for training, validating, testing and saving model'''
import modules as m
import datasets as ds
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import pandas as pd
import copy
from sklearn.manifold import TSNE
from statistics import fmean

PATH = './best_model.pt'
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

def train_gcn(epoch):
    gcn.train()
    # reset grads
    gcn.zero_grad(set_to_none=True)
    output = gcn(node_features, adj_mtrx)
    # print("index", idx_train)
    print("output", output.shape)
    # print("output train", output[idx_train].shape)
    # print(output[idx_train])
    # print("labels", labels.shape)
    # print(labels[idx_train])
    loss_fn = nn.CrossEntropyLoss()
    training_loss = loss_fn(output[train_ids], labels[train_ids])
    # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    # acc_train = (labels[idx_train] == output[idx_train].max(1).item() / labels[idx_train].size(0))
    training_acc = pred_accuracy(output[train_ids], labels[train_ids])
    training_loss.backward()
    optimiser.step()

    gcn.eval()
    output = gcn(node_features, adj_mtrx)

    valid_loss = loss_fn(output[val_ids], labels[val_ids])
    
    valid_acc = pred_accuracy(output[val_ids], labels[val_ids])
    val_loss.append(valid_loss.item())
    train_loss.append(training_loss.item())
    val_acc.append(valid_acc.item())
    train_acc.append(training_loss.item())

    # print("val", output[idx_val])
    # print("train", output[idx_train])
    print(f'Epoch: {epoch + 1}',
          f'loss_train: {training_loss.item()}',
          f'acc_train: {training_acc.item()}',
          f'loss_val: {valid_loss.item()}',
          f'acc_val: {valid_acc.item()}')

def test_gcn(output):
    # gcn.eval()
    output = gcn(node_features, adj_mtrx)
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(output.detach().numpy())
    # print((tsne_results[:, 0]))
    print(pd.DataFrame(tsne_results))
    print(pd.DataFrame(labels, columns=["labels"]))
    with_labels = pd.concat([pd.DataFrame(tsne_results), 
                             pd.DataFrame(labels, columns=["labels"])], axis=1)
    print(with_labels.describe())


    for i in range(5):
      plt.scatter(with_labels.loc[with_labels.labels == i, 0], 
                  with_labels.loc[with_labels.labels == i, 1], 
                  label=i)
      
      loss_fn = nn.CrossEntropyLoss()
      loss_test = loss_fn(output[test_ids], labels[test_ids])
      acc_test = pred_accuracy(output[test_ids], labels[test_ids])

    plt.xlabel("TSNE 0")
    plt.ylabel("TSNE 1")
    plt.title(f"tSNE Embedding from GCN Applied to Facebook Dataset")
    plt.legend([0, 1, 2, 3])
    plt.show()
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

#### TRAIN & SAVE BEST MODEL
best_acc = 0
best_gcn = None
for epoch in range(300):
    train_gcn(epoch)

    if val_acc[epoch] > best_acc:
      best_gcn = copy.deepcopy(gcn)

torch.save(best_gcn, PATH)
test_ = torch.load(f'{PATH}')
test_.eval()

plt.title("Training and Validation Loss")
plt.plot(val_loss,label="val")
plt.plot(train_loss,label="train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

plt.title("Training and Validation Accuracy")
plt.plot(val_acc,label="val")
plt.plot(train_acc,label="train")
plt.xlabel("Epochs")
plt.ylabel("Accuracy ")
plt.legend()
plt.show()

# Testing
print(test_.features)
test_gcn(test_)