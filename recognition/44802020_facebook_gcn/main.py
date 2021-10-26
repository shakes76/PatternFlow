from typing import Any

import keras.activations

import model
import tensorflow as tf
import tensorflow.keras.optimizers as op
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Softmax, Input
from tensorflow.keras import losses, layers, models, activations
import scipy.sparse as spr
import numpy as np


# Constants
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

    features = data['features']
    edges = data['edges']
    target = data['target']

    x_train = features
    y_train = features

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

    # Construct Model
    my_model = Sequential()
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)
    # activation_fn = activations.relu()
    a_bar = model.coo_matrix_to_sparse_tensor(a_bar)

    my_model.add(Input(shape=tf.shape(feats)))

    my_model.add(model.FaceGCNLayer(adj_m=a_bar))
    my_model.add(model.FaceGCNLayer(adj_m=a_bar))
    my_model.add(model.FaceGCNLayer(adj_m=a_bar))

    my_model.add(Dense(4, activation='softmax'))

    my_model.compile(optimizer='adam', loss=loss_fn)
    my_model.fit(feats,
                 labels,
                 epochs=5)

    print(my_model.summary())


if __name__ == '__main__':
    main()



