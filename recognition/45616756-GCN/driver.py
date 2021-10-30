import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, eye, diags
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


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
    train_target, test_target = train_test_split(
        facebook_target, train_size=0.20, test_size=None, stratify=facebook_target
    )
    validation_target, test_target = train_test_split(
        test_target, train_size=0.20, test_size=None, stratify=test_target
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


if __name__ == '__main__':
    main()
