import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


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


if __name__ == '__main__':
    main()
