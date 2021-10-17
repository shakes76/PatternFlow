import tensorflow as tf


def get_nodes(edges):
    nodes = set()
    for edge in edges:
        nodes.add(edge[0])
        nodes.add(edge[1])
    return nodes


def get_adj_mat(edges):
    nodes = get_nodes(edges)
    size = len(nodes)
    adj_mat = [[0. for i in range(size)] for j in range(size)]
    for edge in edges:
        adj_mat[edge[0]][edge[1]] = 1.
        adj_mat[edge[1]][edge[0]] = 1.
    for i in range(size):
        adj_mat[i][i] = 1.
    return tf.constant(adj_mat)


def get_deg_mat(adj_mat):
    size = len(adj_mat)
    degrees = tf.reduce_sum(adj_mat, 1)
    deg_mat = tf.linalg.diag(degrees)
    return tf.constant(deg_mat)

