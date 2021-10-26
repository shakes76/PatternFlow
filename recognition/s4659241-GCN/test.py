"""
code for testing the algorithm
by: Kexin Peng, 4659241
"""
import numpy as np
import scipy.sparse as sp
import torch.optim as optim
from algorithm import *


hidden = 32
hidden1 = 64
hidden2 = 32
dropout_rate = 0.5
decay = 5e-4
epochs = 200
learning_rate = 0.01

# load data
facebook = np.load('facebook.npz')
edges = facebook['edges']
features = facebook['features']
target = facebook['target']

num_classes = len(np.unique(target))
num_nodes = features.shape[0]
num_features = features.shape[1]
num_edges = edges.shape[0]

def normalize(matrix):
    row_sum = np.array(matrix.sum(1))
    # 1/sum
    sum_inv = np.power(row_sum, -1).flatten() 
    # if it's infinite, change to 0
    sum_inv[np.isinf(sum_inv)] = 0.   
    # build diagonal matrix
    diag_mat = sp.diags(sum_inv)
    # normalized matrix: D^-1 * matrix
    normalized = diag_mat.dot(matrix) 
    return normalized

# normalize features and transform data to tensor
norm_features = normalize(features)
norm_features = torch.from_numpy(norm_features)
target = torch.from_numpy(target)

# Adjacency matrix A-- n*n
edge_data = np.ones(num_edges)
row = edges[:, 0]
col = edges[:, 1]
adj_matrix = sp.coo_matrix((edge_data, (row, col)),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)
# A+I add identity matrix
new_matrix = adj_matrix + sp.eye(adj_matrix.shape[0])
# normalize the matrix
new_matrix = normalize(new_matrix) 

# transform scipy sparse matrix to torch tensor
def sparse_to_tensor(sparse):
    sparse = sparse.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse.row, sparse.col)).astype(np.int64))
    values = torch.from_numpy(sparse.data)
    shape = torch.Size(sparse.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

new_matrix = sparse_to_tensor(new_matrix)

# split data into train:validation:test = 2:2:6
train_index = torch.LongTensor(range(int(num_nodes/10*2)))
val_index = torch.LongTensor(range(int(num_nodes/10*2),int(num_nodes/10*4)))
test_index = torch.LongTensor(range(int(num_nodes/10*4),num_nodes))

# initiate model
# 2 layers mode:
# model = GCN(n_feature = num_features, n_hidden = hidden, n_class = num_classes, dropout = 0.5)
# 3 layers model
model = GCN_3l(n_feature = num_features, n_hidden1 = hidden1, n_hidden2 = hidden2, n_class = num_classes, 
    dropout = dropout_rate)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=decay)
model.train()

def test(index):
    model.eval()
    with torch.no_grad():
        output = model(norm_features,new_matrix)
        test_output = output[index]
        predict_y = test_output.max(1)[1]
        accuarcy = torch.eq(predict_y, target[index]).float().mean()
    return accuarcy, test_output.cpu().numpy(), target[index].cpu().numpy()

loss_history = []
train_acc_history = []
val_acc_history = []
train_y = target[train_index]
for epoch in range(epochs):
    outputs = model(norm_features,new_matrix) 
    train_out = outputs[train_index]
    # calculate loss
    loss = criterion(train_out, train_y)
    optimizer.zero_grad()
    loss.backward() 
    optimizer.step()
    # calculate accuracy on train and validation set
    train_acc, _, _ = test(train_index)
    val_acc, _, _ = test(val_index) 
    # store the loss and accuracy
    loss_history.append(loss.item())
    train_acc_history.append(train_acc.item())
    val_acc_history.append(val_acc.item())
    print("Epoch {:03d}: Loss {:.4f}, TrainAcc {:.4}, ValAcc {:.4f}".format(
        epoch, loss.item(), train_acc.item(), val_acc.item()))

# Test dataset
test_acc, _, _ = test(test_index)
print("Test Accuracy: {:.4f}".format(float(test_acc.numpy())))


# code for TSNE embeddings plot
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# def plot_embedding(data, label, title):
#     x_min, x_max = np.min(data, 0), np.max(data, 0)
#     data = (data - x_min) / (x_max - x_min)

#     fig = plt.figure()
#     ax = plt.subplot(111)
#     for i in range(data.shape[0]):
#         plt.text(data[i, 0], data[i, 1], str(label[i]),
#                  color=plt.cm.Set2(label[i] / 10.),
#                  fontdict={'weight': 'bold', 'size': 9})
#     plt.xticks([])
#     plt.yticks([])
#     plt.title(title)
#     return fig

# pca_50 = PCA(n_components=50) # change to 90 if reduce dimensions to 90
# pca_result_50 = pca_50.fit_transform(features)
# tsne = TSNE(perplexity=30, n_components=2, init='pca')
# result = tsne.fit_transform(pca_result_50)
# result = tsne.fit_transform(features) # without using PCA first
# fig = plot_embedding(transfromed, facebook['target']+1,'t-SNE')
# plt.show(fig)
