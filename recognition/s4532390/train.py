import torch
from torch.nn import CrossEntropyLoss
from modules import GCN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

NUM_CLASSES = 4
EPOCH_NUM = 20

def run_model(features, adjacency_matrix, targets, tsne_plot=False):
    """
    Trains a GCN and tests it against the given targets
    Also plots the TSNE mapping if activated
    """
    num_pages = len(features)
    feature_dim = features.shape[1]

    # Train, val, test split = 6, 2, 2
    train_index = torch.LongTensor(range(int(num_pages/10*6)))
    val_index = torch.LongTensor(range(int(num_pages/10*6), int(num_pages/10*8)))
    test_index = torch.LongTensor(range(int(num_pages/10*8), num_pages))

    # Activate CUDA if possible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = GCN(feature_dim, NUM_CLASSES).to(device)

    # Set optimizer and loss funcion
    loss_function = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    val_accuracies = []
    train_accuracies = []
    loss_list = []

    ## Training the model ##
    print("Training model...")
    model.train()
    for epoch in range(EPOCH_NUM):
        optimizer.zero_grad()
        out = model(features, adjacency_matrix)
        train_out = out[train_index]
        loss = loss_function(train_out, targets[train_index])
        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()
        model.eval()

        # Validation and training accuracy
        pred = model(features, adjacency_matrix).argmax(dim=1)
        correct_train = (pred[train_index] == targets[train_index]).sum()
        acc_train = correct_train / len(targets[train_index])
        train_accuracies.append(acc_train)

        correct_val = (pred[val_index] == targets[val_index]).sum()
        acc_val = correct_val / len(targets[val_index])
        val_accuracies.append(acc_val)
        print("Epoch {:d}: Loss {:.4f}, Training Accuracy {:.4}, Validation Accuracy {:.4f}".format(
                epoch + 1, loss.item(), acc_train, acc_val))
    
    print("Finished training")

    plt.figure(figsize=(8,8))
    plt.plot(val_accuracies)

    ## Testing ##
    model.eval()
    pred = model(features, adjacency_matrix).argmax(dim=1)
    correct = (pred[test_index] == targets[test_index]).sum()
    acc = correct / len(targets[test_index])
    print(f'Test Accuracy: {acc:.4f}')

    ## Plot TSNE Mapping ##
    if tsne_plot:
        pred = model(features, adjacency_matrix)
        np_pred = pred.cpu().detach().numpy()
        np_targets = targets.cpu().detach().numpy()
        one_hot_targets = np.zeros((num_pages, NUM_CLASSES))
        one_hot_targets[np.arange(num_pages), np_targets] = 1
        plot_tsne_mapping(np_pred, one_hot_targets)

def plot_tsne_mapping(pred, targets):

    pred_tsne = TSNE(n_components=2).fit_transform(pred)

    color_map = np.argmax(targets, axis=1)

    plt.figure(figsize=(8,8))

    for i in range(NUM_CLASSES):
        indices = np.where(color_map == i)
        indices = indices[0]
        plt.scatter(pred_tsne[indices, 0], pred_tsne[indices, 1], label=i)
    
    plt.legend()
    plt.show()
    