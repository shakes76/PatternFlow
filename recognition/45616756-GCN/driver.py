import numpy as np
import torch
import torch.optim as optim
from scipy.sparse import coo_matrix, csr_matrix, eye, diags
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from model import GCN


def main():
    # Load Facebook dataset
    data = np.load('./data/facebook.npz')

    # Create an adjacency matrix representation based on the edges
    facebook_edges = data['edges']
    adjacency_matrix = np.zeros((22470, 22470), dtype='float32')
    for edge in facebook_edges:
        adjacency_matrix[edge[0]][edge[1]] = 1
    adjacency_matrix = coo_matrix(adjacency_matrix)

    # Nodes features
    facebook_features = data['features']
    facebook_features = csr_matrix(facebook_features)

    # Convert each categorical value (One-hot encoding)
    facebook_target = data['target']
    lb = preprocessing.LabelBinarizer()
    facebook_target = lb.fit_transform(facebook_target)

    # Split the target (20:20:60)
    facebook_train_target, facebook_test_target = train_test_split(
        facebook_target, train_size=0.20, shuffle=False
    )
    facebook_validation_target, facebook_test_target = train_test_split(
        facebook_test_target, train_size=0.20, shuffle=False
    )

    # Normalize the adjacency matrix
    a_tilde = adjacency_matrix + eye(22470, dtype='float32')  # adjacency matrix + self-loop
    d = diags(np.array(a_tilde.sum(axis=1)).flatten())  # degree matrix
    degrees_inverse = np.power(d.diagonal(), -1)
    d_inverse = diags(degrees_inverse)  # degree matrix inverse
    adjacency_matrix = d_inverse.dot(a_tilde).tocoo()

    # Normalize the adjacency matrix (Ver.2)
    # A_tilde = adjacency_matrix + np.eye(22470)
    # D_tilde = np.matrix(np.diag(np.array(np.sum(A_tilde, axis=0))[0]))
    # D_tilde_invroot = np.linalg.inv(sqrtm(D_tilde))
    # A_hat = np.matmul(np.matmul(A_tilde, D_tilde_invroot), D_tilde_invroot)

    # Normalize the features matrix
    d = diags(np.array(facebook_features.sum(axis=1)).flatten())  # degree matrix
    degrees_inverse = np.power(d.diagonal(), -1)
    d_inverse = diags(degrees_inverse)  # degree matrix inverse
    facebook_features = d_inverse.dot(facebook_features)

    # Convert to tensor
    facebook_features = torch.FloatTensor(np.array(facebook_features.todense()))
    facebook_train_target = torch.LongTensor(np.where(facebook_train_target)[1])
    facebook_validation_target = torch.LongTensor(np.where(facebook_validation_target)[1])
    facebook_test_target = torch.LongTensor(np.where(facebook_test_target)[1])
    adjacency_matrix = torch.sparse.FloatTensor(
        torch.LongTensor(np.vstack((adjacency_matrix.row, adjacency_matrix.col))),
        torch.FloatTensor(adjacency_matrix.data),
        torch.Size(adjacency_matrix.shape)
    )

    # Print output
    print('facebook_features:', facebook_features)
    print('facebook_train_target:', facebook_train_target.size())
    print('facebook_validation_target:', facebook_validation_target.size())
    print('facebook_test_target:', facebook_test_target.size())
    print('adjacency_matrix:', adjacency_matrix)

    model = GCN(input_size=facebook_features.shape[1],
                hidden_size=16,
                num_classes=4,
                dropout=0.5)
    optimizer = optim.Adam(model.parameters(),
                           lr=0.01,
                           weight_decay=5e-4)

    print('model', model)
    print('optimizer', optimizer)


if __name__ == '__main__':
    main()
