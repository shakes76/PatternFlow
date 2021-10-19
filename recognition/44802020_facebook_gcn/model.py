import numpy as np
import scipy
import scipy.sparse as spr
import tensorflow as tf


def get_neighbours(self, node):
    print(node)


def denseNDArrayToSparseTensor(arr):
    idx = np.where(arr != 0.0)
    return tf.SparseTensor(np.vstack(idx).T, arr[idx], arr.shape)


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def coo_matrix_to_sparse_tensor(coo):
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def Model(input_data):

    # This is the model
    page_one = [0, 0, 1, 2, 0]
    page_two = [4, 2, 3, 3, 1]

    identity = [0, 1, 2, 3, 4]

    page_one += identity
    page_two += identity

    print("ID")
    print(page_one)

    ones = tf.ones_like(page_one)

    feats = [[0.3, 2.2, -1.7],
             [4., -1.3, -1.2],
             [0.3, 2.2, 0.5],
             [0.5, 0.7, -3.5],
             [2.0, 5.2, -0.6]
             ]

    # Pick a Node V
    # node = input_data[0]

    # Construct Adjacency matrix
    A = spr.coo_matrix((ones, (page_one, page_two)))

    print("Running Model")
    print(A)
    print("=====  =====")
    print(A.data)
    print(A.dtype)
    print(A.shape)
    print(A.col)
    print(A.row)

    A_spr = coo_matrix_to_sparse_tensor(A)

    # feats_sparse = spr.csr_matrix(feats)
    # A_sp = tf.sparse.from_dense(A)
    # F = tf.sparse.from_dense(feats)
    # print(F.shape)

    print("===== Result =====")


    # newA = tf.sparse.SparseTensor(A)
    #
    res = tf.sparse.sparse_dense_matmul(tf.cast(A_spr, float), feats)
    print(res.dtype)
    print(res)



