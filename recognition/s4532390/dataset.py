import numpy as np
import tensorflow as tf
from sklearn import model_selection
from sklearn import preprocessing
from torch.nn import CrossEntropyLoss
from torch.nn.parameter import Parameter
from torch_geometric.data import Data
import torch
import scipy.sparse
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.modules.module import Module

NUM_CLASSES = 4

def parse_data(file_path, test_size):
    """
    
    """
    # Extracts the data from the npz file
    with np.load(file_path) as data:

        # 22470 integers (0, 1, 2, 3) corresponding to the categories - politicians, governmental organizations, television shows and companies
        targets = data['target']
        targets = torch.from_numpy(targets)

        # 22470 vectors - each with 128 features
        features = data['features']
        features = torch.from_numpy(features)
        features = F.normalize(features, p=1, dim=1)
        # features = preprocessing.normalize(features, axis=1, norm='l1')
        

        # 342004 size edge list
        edges = data['edges']

        # Constructing adjaceny matrix
        num_pages = len(features)
        num_edges = len(edges)
        feature_dim = features.shape[1]
        edge_in = edges[:,0]
        edge_out = edges[:,1]

        coo_matrix = scipy.sparse.coo_matrix((np.ones(num_edges), (edge_in, edge_out)),
                                                    shape=(num_pages, num_pages),
                                                    dtype=np.float32)
        # Add identity matrix as each page connects to itself
        coo_matrix += scipy.sparse.eye(coo_matrix.shape[0])
        # Normalise the matrix
        # coo_matrix = F.normalize(coo_matrix, p=1, dim=1)
        coo_matrix = preprocessing.normalize(coo_matrix, axis=1, norm='l1')

        sparse = coo_matrix.tocoo().astype(np.float32)
        # indices = torch.from_numpy(np.vstack((sparse.row, sparse.col)).astype(np.int64))
        # values = torch.from_numpy(sparse.data)
        # data = Data(x=values, edge_index=indices)

        # Convert to sparse tensor TODO CHANGE THIS CODE
        def convert_to_adj_tensor(input):

            input = input.tocoo().astype(np.float32)

            indices = torch.from_numpy(np.vstack((input.row, input.col)).astype(np.int64))
            values = torch.from_numpy(input.data)
            shape = torch.Size(input.shape)
            return torch.sparse_coo_tensor(indices, values, shape)

        adjacency_matrix = convert_to_adj_tensor(coo_matrix)

        # Train, val, test split = 6, 2, 2
        train_index = torch.LongTensor(range(int(num_pages/10*6)))
        val_index = torch.LongTensor(range(int(num_pages/10*6), int(num_pages/10*8)))
        test_index = torch.LongTensor(range(int(num_pages/10*8), num_pages))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GCN(feature_dim, NUM_CLASSES).to(device)

        loss_function = CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()

        for epoch in range(200):
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



# What data shape does it want? How to use the 128 dim features?
# How to split dataset if it is a single graph?
# Why make a custom GCNConv layer? Use custom
# Embeddings? See shakes code
# Semi-supervised? Fully is fine

class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adjacency_matrix, active=True):
        support = torch.mm(features, self.weight)
        output = torch.spmm(adjacency_matrix, support)
        if active:
            output = F.relu(output)
        return output


class GCN(torch.nn.Module):
    def __init__(self, in_features, out_classes):
        super().__init__()
        self.conv1 = GNNLayer(in_features, 16)
        self.conv2 = GNNLayer(16, out_classes)

    def forward(self, x, adjacency_matrix):
        x = self.conv1(x, adjacency_matrix)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, adjacency_matrix)

        return F.log_softmax(x, dim=1)



parse_data('recognition\\s4532390\\res\\facebook.npz', 0.2)

    



    # data[]
# data = np.load('facebook.npz')

# print(data)