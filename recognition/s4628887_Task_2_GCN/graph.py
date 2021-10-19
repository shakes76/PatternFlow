import tensorflow as tf


def get_nodes(edges):
    """
    Get the nodes of a graph network from the array of edges
    :param edges: array of edges
    :return: a set of nodes
    """
    nodes = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    return nodes


def get_adj_mat(edges):
    """
    Get the adjacency matrix of a graph network from the array of edges
    :param edges: array of edges
    :return: an adjacency matrix as a tensor
    """
    nodes = get_nodes(edges)
    size = len(nodes)
    adj_mat = [[0. for i in range(size)] for j in range(size)]
    for edge in edges:
        adj_mat[edge[0]][edge[1]] = 1.
        adj_mat[edge[1]][edge[0]] = 1.
    for i in range(size):
        adj_mat[i][i] = 1.
    return tf.constant(adj_mat)
    
    
def get_half_norm_deg_mat(adj_mat):
    """
    Compute the half normalised degree matrix from the adjacency matrix
    :param adj_mat: an adjacency matrix as a tensor
    :return: a half normalised degree matrix
    """
    degrees = tf.reduce_sum(adj_mat, 1)
    half_norm_degrees = tf.math.sqrt(tf.math.reciprocal(degrees))
    half_norm_deg_mat = tf.linalg.diag(half_norm_degrees)
    return tf.constant(half_norm_deg_mat)


def get_adj_mat_hat(adj_mat):
    """
    Normalise the adjacency matrix
    :param adj_mat: an adjacency matrix as a tensor
    :return: Normalised adjacency matrix as (\D^{-\frac{1}{2}}\A\D^{-\frac{1}{2}}\)
    where D is the degree matrix
    """
    D_half_norm = get_half_norm_deg_mat(adj_mat)
    return tf.linalg.matmul(tf.linalg.matmul(D_half_norm, adj_mat, a_is_sparse=True), D_half_norm, b_is_sparse=True)
