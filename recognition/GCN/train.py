'''contains the source code for training, validating, testing and saving model'''
import modules as m
import dataset as ds
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import pandas as pd
import copy
from sklearn.manifold import TSNE
from dataset import preprocess
from modules import Net 
import torch.optim as optim

MODEL_PATH = './best_model.pt'
N_EPOCHS = 300
N_HID = 20
N_OUTPUT = 4

# Load data
# adj_mtrx, node_features, labels, train_ids, val_ids, test_ids = ds.preprocess()
# node_features = node_features.double()
# adj_mtrx = adj_mtrx.double()
val_loss = []
train_loss = []
val_acc = []
train_acc = []

# ratio of highest probability predictions to labels
def pred_accuracy(pred, labels):
    acc = (pred.max(1)[1] == labels).sum() / len(labels)
    return acc

def train_gcn(epoch, input_gcn, optimiser, node_features, adj_mtrx, labels,
              train_ids, val_ids):
    input_gcn.train()

    # set gradient to none every iteration 
    input_gcn.zero_grad(set_to_none=True)
    output = input_gcn(node_features, adj_mtrx)

    # define loss for backprop
    loss_fn = nn.CrossEntropyLoss()
    training_loss = loss_fn(output[train_ids], labels[train_ids])

    # predict accuracy
    training_acc = pred_accuracy(output[train_ids], labels[train_ids])

    # backpropagate loss 
    training_loss.backward()
    optimiser.step()

    input_gcn.eval()

    # verify model performance on validation set
    output = input_gcn(node_features, adj_mtrx)
    valid_loss = loss_fn(output[val_ids], labels[val_ids])
    valid_acc = pred_accuracy(output[val_ids], labels[val_ids])
    val_loss.append(valid_loss.item())
    train_loss.append(training_loss.item())
    val_acc.append(valid_acc.item())
    train_acc.append(training_acc.item())

    print(f'Epoch: {epoch + 1}',
          f'loss_train: {training_loss.item()}',
          f'acc_train: {training_acc.item()}',
          f'loss_val: {valid_loss.item()}',
          f'acc_val: {valid_acc.item()}')

#train on set epoch number and save best performing mdoel
def run_training(training=True):
    # load in adjacency matrix, labels and indices
    adj_mtrx, node_features, labels, train_ids, val_ids, test_ids = preprocess()
    node_features = node_features.double()
    adj_mtrx = adj_mtrx.double()

    gcn = Net(node_features.shape[1], N_HID, N_OUTPUT)
    optimiser = optim.Adam(gcn.parameters())
    
    # keep track of best performing model
    best_acc = 0
    best_gcn = None

    for epoch in range(N_EPOCHS):
        train_gcn(epoch, gcn, optimiser, node_features, 
        adj_mtrx, labels, train_ids, val_ids)

        if val_acc[epoch] > best_acc:
            best_gcn = copy.deepcopy(gcn)

    torch.save(best_gcn.state_dict(), MODEL_PATH)

    plt.title("Training and Validation Loss")
    plt.plot(val_loss,label="val")
    plt.plot(train_loss,label="train")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    print(train_acc)
    plt.title("Training and Validation Accuracy")
    plt.plot(val_acc,label="val")
    plt.plot(train_acc,label="train")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy ")
    plt.legend()
    plt.show()

    input_gcn = best_gcn(node_features, adj_mtrx)
    print(best_gcn.features)
    test_gcn(input_gcn[test_ids], labels[test_ids])

# test GCN if in testing mode- else, predict on passed inputs
def test_gcn(input_gcn, labels=[], test=True):
    label_dict = {0:"politicians", 
                  1:"governmental organizations", 
                  2: "television shows", 
                  3:"companies"}
    if test:
        loss_fn = nn.CrossEntropyLoss()
        loss_test = loss_fn(input_gcn, labels)
        acc_test = pred_accuracy(input_gcn, labels)
        print("Test set results:",
            f"loss= {loss_test.item()}), accuracy= {(acc_test.item())}")
    else:
        print(f"Predicted nodes are:")
        for i in range(len(input_gcn.max(1)[1])):
            node = input_gcn.max(1)[1][i].item()
            print(f"{i}. {node}- {label_dict[node]}")

def plot_tsne(embeddings, labels):
    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(embeddings.detach().numpy())
    with_labels = pd.concat([pd.DataFrame(tsne_results), 
                            pd.DataFrame(labels, columns=["labels"])], axis=1)
    for i in range(5):
        plt.scatter(with_labels.loc[with_labels.labels == i, 0], 
                with_labels.loc[with_labels.labels == i, 1], 
                label=i)

    plt.xlabel("TSNE 0")
    plt.ylabel("TSNE 1")
    plt.title(f"tSNE Embedding from GCN Applied to Facebook Dataset")
    plt.legend([0, 1, 2, 3])
    plt.show()

# comment function call if running predict.py
run_training()
