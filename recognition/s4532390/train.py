import torch
from torch.nn import CrossEntropyLoss
from modules import GCN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

NUM_CLASSES = 4
EPOCH_NUM = 20

def run_model(features, adjacency_matrix, targets):
    """
    
    """

    num_pages = len(features)
    feature_dim = features.shape[1]

    # Train, val, test split = 6, 2, 2
    train_index = torch.LongTensor(range(int(num_pages/10*6)))
    val_index = torch.LongTensor(range(int(num_pages/10*6), int(num_pages/10*8)))
    test_index = torch.LongTensor(range(int(num_pages/10*8), num_pages))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(feature_dim, NUM_CLASSES).to(device)

    loss_function = CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()


    val_accuracies = np.ones(EPOCH_NUM)
    test_accuracies = np.ones(EPOCH_NUM)

    for epoch in range(EPOCH_NUM):
        optimizer.zero_grad()
        out = model(features, adjacency_matrix)
        train_out = out[train_index]
        loss = loss_function(train_out, targets[train_index])
        loss.backward()
        optimizer.step()

        model.eval()
        pred = model(features, adjacency_matrix).argmax(dim=1)
        correct = (pred[val_index] == targets[val_index]).sum()
        acc = correct / len(targets[val_index])
        print(f'Validation Accuracy: {acc:.4f}')

    model.eval()
    pred = model(features, adjacency_matrix).argmax(dim=1)
    correct = (pred[test_index] == targets[test_index]).sum()
    acc = correct / len(targets[test_index])
    print(f'Accuracy: {acc:.4f}')

    pred = model(features, adjacency_matrix)

    np_pred = pred.cpu().detach().numpy()

    np_targets = targets.cpu().detach().numpy()
    one_hot_targets = np.zeros((num_pages, NUM_CLASSES))
    one_hot_targets[np.arange(num_pages), np_targets] = 1


    pred_tsne = TSNE(n_components=2).fit_transform(np_pred)

    color_map = np.argmax(one_hot_targets, axis=1)

    plt.figure(figsize=(8,8))

    for i in range(NUM_CLASSES):
        indices = np.where(color_map == i)
        indices = indices[0]
        plt.scatter(pred_tsne[indices,0], pred_tsne[indices, 1], label=i)
    
    plt.legend()
    plt.show()
    