import torch
from torch.nn import CrossEntropyLoss
from modules import GCN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Constants
NUM_CLASSES = 4
EPOCH_NUM = 200

def run_model(features, adjacency_matrix, targets, tsne_plot=False):
    """
    Trains a GCN and tests it against the given targets
    Also plots the TSNE embedding if activated
    Inputs:
        features - 2d tensor of feature vectors (each 128 dim)
        adjacency_matrix - coordinate format sparse tensor of graph adjacency matrix
        targets - 1d tensor of integers (0 - 3) representing politicians, government organisations, television shows and companies pages
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
    train_loss_list = []
    val_loss_list = []

    ## Training the model ##
    print("Training model...")
    model.train()
    for epoch in range(EPOCH_NUM):
        # Optimising model
        optimizer.zero_grad()
        out = model(features, adjacency_matrix)
        train_loss = loss_function(out[train_index], targets[train_index])
        train_loss_list.append(train_loss.item())
        train_loss.backward()
        optimizer.step()
        model.eval()

        # Validation loss
        val_loss = loss_function(out[val_index], targets[val_index])
        val_loss_list.append(val_loss.item())

        # Validation and training accuracy
        pred = model(features, adjacency_matrix).argmax(dim=1)
        correct_train = (pred[train_index] == targets[train_index]).sum()
        acc_train = correct_train / len(targets[train_index])
        train_accuracies.append(acc_train)

        correct_val = (pred[val_index] == targets[val_index]).sum()
        acc_val = correct_val / len(targets[val_index])
        val_accuracies.append(acc_val)
        print("Epoch {:d}: Loss {:.4f}, Validation Loss {:.4f}, Training Accuracy {:.4}, Validation Accuracy {:.4f}".format(
                epoch + 1, train_loss, val_loss, acc_train, acc_val))
    
    print("Finished training")

    plt.figure(figsize=(8,8))
    plt.plot(range(len(val_accuracies)), val_accuracies, label="Validation Accuracy")
    plt.plot(range(len(train_accuracies)), train_accuracies, label="Training Accuracy")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy during training")

    plt.figure(figsize=(8,8))
    plt.plot(range(len(train_loss_list)), train_loss_list, label="Training Loss")
    plt.plot(range(len(val_loss_list)), val_loss_list, label="Validation Loss")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss during training")

    ## Testing ##
    model.eval()
    pred = model(features, adjacency_matrix).argmax(dim=1)
    correct = (pred[test_index] == targets[test_index]).sum()
    acc = correct / len(targets[test_index])
    print(f'Test Accuracy: {acc:.4f}')

    ## Plot TSNE Mapping ##
    if tsne_plot:
        print("Running T-Distributed Stochastic Embedding...")

        # Converts predictions into numpy format
        pred = model(features, adjacency_matrix)
        np_pred = pred.cpu().detach().numpy()

        # Gets targets in one-hot encoded format
        np_targets = targets.cpu().detach().numpy()
        one_hot_targets = np.zeros((num_pages, NUM_CLASSES))
        one_hot_targets[np.arange(num_pages), np_targets] = 1
        plot_tsne_mapping(np_pred, one_hot_targets)
        print("Finished embedding")


def plot_tsne_mapping(pred, targets):
    """
    Takes in the prediction values from the trained model and the targets in One-hot encoded format
    Using T-Distributed Stochastic Embedding to reduce the dimensions so it can be visualised in a human-readable format
    """
    # Reducing down to 2 dimensions
    pred_tsne = TSNE(n_components=2).fit_transform(pred)

    plt.figure(figsize=(8,8))

    color_map = np.argmax(targets, axis=1)

    # Repeated for each class (4 times)
    for i in range(NUM_CLASSES):
        # Each class has a different colour
        indices = np.where(color_map == i)
        indices = indices[0]

        # Scatter the embeddings on the plot
        plt.scatter(pred_tsne[indices, 0], pred_tsne[indices, 1], label=i)
    plt.title("T-Distributed Stochastic Embedding Visualised")
    plt.legend()