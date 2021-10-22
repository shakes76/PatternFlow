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


class MyLayer(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(MyLayer, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

        self.kernal = 9

    def build(self):
        self.kernal = 9

    def call(self, inputs):

        return tf.matmul(inputs, self.w) + self.b


class FacebookGCN(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(FacebookGCN, self).__init__(name='')
        filters1, filters2, filters3 = filters

        self.graph_layer = MyLayer()
        self.softmax = tf.keras.layers.Softmax()

    def call(self, input_tensor, training=False):

        x = MyLayer(input_tensor)
        x = self.softmax(x)

        x = MyLayer(x)
        x = self.softmax(x)

        x = lyr.Dense()

        x += input_tensor
        return tf.nn.relu(x)
