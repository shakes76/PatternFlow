from typing import Any

import keras.activations
import scipy.sparse

import myGraphModel
import tensorflow as tf
import tensorflow.keras.optimizers as op
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Softmax, Input
from tensorflow.keras import losses, layers, models, activations
import scipy.sparse as spr
import numpy as np


# Constants
def coo_matrix_to_sparse_tensor(coo):
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)

def summarise_data(data, aspect):
    print(f"===== {aspect} =====")
    aspect_d = data[aspect]
    print(aspect_d.shape)  # (22 470, 128)
    print(aspect_d)
    print(type(aspect_d))
    print(type(aspect_d[0]))
    print(aspect_d[0])
    print("====================")


def main():
    print("Tensorflow version:", tf.__version__)
    print("Numpy version:", np.__version__)

    # file_path = r"C:\Users\johnv\Documents\Code Projects\Pattern Recognition\facebook.npz"

    file_path = r"C:\Users\johnv\Documents\University\COMP3710\Pattern Flow Project\facebook.npz"

    # Load in Data
    data = np.load(file_path)

    # print(data.files)

    # features = data['features']
    # edges = data['edges']
    # target = data['target']

    # summarise_data(data, 'features')
    # summarise_data(data, 'edges')
    # summarise_data(data, 'target')

    # There are 22 470 Pages
    # Each with 128 features
    # Each falls into 1 of 4 categories
    # There are 342 004 Edges between the pages

    # ================== TEST MODEL ========================
    # This is the model
    page_one = [0, 0, 1, 2, 0, 4, 2, 3, 3, 1]
    page_two = [4, 2, 3, 3, 1, 0, 0, 1, 2, 0]

    name = [0, 1, 2, 3, 4]

    page_one += name
    page_two += name

    ones = tf.ones_like(page_one)

    feats = [[0.3, 2.2, -1.7],
             [4., -1.3, -1.2],
             [0.3, 2.2, 0.5],
             [0.5, 0.7, 3.5],
             [2.0, 5.2, 0.6]
             ]

    labels = [0,
             0,
             1,
             3,
             2
              ]

    # Construct Adjacency matrix
    a_bar = spr.coo_matrix((ones, (page_one, page_two)))
    a_bar.setdiag(1)
    a_bar = coo_matrix_to_sparse_tensor(a_bar)

    # ================== REAL MODEL ========================
    # Adjacency Matrix
    page_one = data['edges'][:, 0]
    page_two = data['edges'][:, 1]

    ones = tf.ones_like(page_one)

    a_bar = spr.coo_matrix((ones, (page_one, page_two)))
    a_bar.setdiag(1)
    print("A_Bar 1")
    print(a_bar)
    print(a_bar.shape)
    print(type(a_bar))
    # Normalize
    row_sum = np.array(a_bar.sum(1))
    d_inv_sqr = np.power(row_sum, -0.5).flatten()
    d_inv_sqr[np.isinf(d_inv_sqr)] = 0
    d_mat_inv_sqrt = scipy.sparse.diags(d_inv_sqr)
    a_bar = a_bar.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    print("A_Bar 2")
    print(a_bar)
    print(a_bar.shape)
    print(type(a_bar))
    a_bar = coo_matrix_to_sparse_tensor(a_bar)

    # Features
    # print("Feats 1")
    # print(feats)
    # print(tf.shape(feats))
    # print(type(feats))
    feats = tf.convert_to_tensor(data['features'])
    # print("Feats 2")
    # print(feats)
    # print(tf.shape(feats))
    # print(type(feats))

    # Labels
    print("Labels 1")
    print(labels)
    print(tf.shape(labels))
    print(type(labels))
    labels = tf.convert_to_tensor(data['target'])
    print("Labels 2")
    print(labels)
    print(tf.shape(labels))
    print(type(labels))
    # ================== REAL MODEL DONE ===================

    # Construct Model
    my_model = Sequential()
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)
    # activation_fn = activations.relu()

    print("Shape 1")
    print(tf.shape(feats))
    print("Shape 2")
    print(tf.Tensor.get_shape(feats))
    print("Shape 3")
    print(tf.shape(feats).get_shape())

    my_model.add(Input(shape=tf.Tensor.get_shape(feats)))
    #my_model.add(Input(batch_shape=tf.shape(feats)))

    my_model.add(myGraphModel.FaceGCNLayer(adj_m=a_bar))
    # my_model.add(myGraphModel.FaceGCNLayer(adj_m=a_bar))
    # my_model.add(myGraphModel.FaceGCNLayer(adj_m=a_bar))

    my_model.add(Dense(4, activation='softmax'))

    my_model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    my_model.fit(feats,
                 labels,
                 epochs=1000,
                 batch_size=22470
                 )

    print(my_model.summary())


if __name__ == '__main__':
    main()



