import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
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


if __name__ == '__main__':
    main()
