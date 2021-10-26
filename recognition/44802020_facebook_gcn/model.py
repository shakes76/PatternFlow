import keras.initializers.initializers_v1
import numpy as np
import scipy
import scipy.sparse as spr
import tensorflow as tf
import tensorflow.keras.layers as lyr
import tensorflow.keras as ks


@tf.function
def printIncident(incident):
    print("MyFunc")
    print(incident)
    return incident


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


def test_layer(input_data):

    # This is the model
    page_one = [0, 0, 1, 2, 0, 4, 2, 3, 3, 1]
    page_two = [4, 2, 3, 3, 1, 0, 0, 1, 2, 0]

    name = [0, 1, 2, 3, 4]

    page_one += name
    page_two += name

    print("ID")
    print(page_one)

    ones = tf.ones_like(page_one)

    feats = [[0.3, 2.2, -1.7],
             [4., -1.3, -1.2],
             [0.3, 2.2, 0.5],
             [0.5, 0.7, -3.5],
             [2.0, 5.2, -0.6]
             ]

    # Construct Adjacency matrix
    a_bar = spr.coo_matrix((ones, (page_one, page_two)))
    a_bar.setdiag(1)

    print(a_bar.toarray())

    print("Running Model")
    print("===== Sparse =====")
    a_bar_spr = coo_matrix_to_sparse_tensor(a_bar)

    print("===== Result =====")
    ax = tf.sparse.sparse_dense_matmul(tf.cast(a_bar_spr, float), feats)
    print(ax.dtype)

    print("AX")
    print(ax)
    print("AX - post ReLu")
    print(tf.nn.relu(ax))
    feats = np.array(feats)

    weights = tf.random.normal(feats.shape, mean=0.0, stddev=1.0, dtype=tf.dtypes.float32, seed=None, name=None)

    print(weights)

    z = ax * weights

    print(z)


class FaceGCNLayer(tf.keras.layers.Layer):
    def __init__(self, adj_m):
        super(FaceGCNLayer, self).__init__()
        self.adj_m = adj_m

    def build(self, input_shape):
        self.weights1 = self.add_weight("weights1",
                                       shape=input_shape[1:],
                                       initializer=keras.initializers.initializers_v1.RandomNormal)

    def call(self, feature_matrix):
        feature_matrix = tf.squeeze(feature_matrix)
        ax = tf.sparse.sparse_dense_matmul(tf.cast(self.adj_m, float), feature_matrix)
        z = ax * self.weights1

        return z

