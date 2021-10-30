import random

import myGraphModel
import tensorflow as tf

from tensorflow.keras import losses
import tensorflow.keras.optimizers as op
import scipy.sparse as spr
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def coo_matrix_to_sparse_tensor(coo):
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.SparseTensor(indices, coo.data, coo.shape)


def normalize_adjacency_matrix(a_bar):
    row_sum = np.array(a_bar.sum(1))
    d_inv_sqr = np.power(row_sum, -0.5).flatten()
    d_inv_sqr[np.isinf(d_inv_sqr)] = 0
    d_mat_inv_sqrt = spr.diags(d_inv_sqr)
    a_bar = a_bar.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return a_bar


def summarise_data(data, aspect):
    print(f"===== {aspect} =====")
    aspect_d = data[aspect]
    print(aspect_d.shape)  # (22 470, 128)
    print(aspect_d)
    print(type(aspect_d))
    print(type(aspect_d[0]))
    print(aspect_d[0])
    print("====================")


def generate_tsne_plot(labels, feats):
    # TSNE
    print("Executing TSNE, this might take a moment...")
    tsne = TSNE(2)
    tsne_data = tsne.fit_transform(feats)
    print(tsne_data.shape)

    plt.figure(figsize=(6, 5))
    plt.scatter(tsne_data[labels == 0, 0], tsne_data[labels == 0, 1], c='b')
    plt.scatter(tsne_data[labels == 1, 0], tsne_data[labels == 1, 1], c='r')
    plt.scatter(tsne_data[labels == 2, 0], tsne_data[labels == 2, 1], c='g')
    plt.scatter(tsne_data[labels == 3, 0], tsne_data[labels == 3, 1], c='y')
    plt.show()


def shuffle(page_one, page_two, feats, labels):
    z = list(zip(page_one, page_two, feats, labels))
    random.shuffle(z)
    page_one, page_two, feats, labels = zip(*z)

    return page_one, page_two, feats, labels


def parse_data(data, train_split, val_split):
    # Adjacency Matrix
    # Split EdgeList into two tensors
    page_one = data['edges'][:, 0]
    page_two = data['edges'][:, 1]
    # Features
    feats = tf.convert_to_tensor(data['features'])
    # Labels
    labels = tf.convert_to_tensor(data['target'])

    # Split Data
    # Data needs to be manually split here because the current implementation requires
    page_one, page_two, feats, labels = shuffle(page_one, page_two, feats, labels)

    page_one = tf.convert_to_tensor(page_one)
    page_two = tf.convert_to_tensor(page_two)
    feats = tf.convert_to_tensor(feats)
    labels = tf.convert_to_tensor(labels)

    # Convert split percentage into integer
    print("SHAPE")
    print(labels.shape[0])
    split_t = int(round(labels.shape[0] * train_split))
    split_v = split_t + int(round(labels.shape[0] * val_split))
    print(f"T:{split_t}, V:{split_v}")

    train_labels, val_labels, test_labels = labels[:split_t], labels[split_t:split_v], labels[split_v:]
    train_feats, val_feats, test_feats = feats[:split_t], feats[split_t:split_v], feats[split_v:]

    # Convert EdgeList to Sparse Adjacency Matrix
    ones = tf.ones_like(page_one)  # Create Ones Matrix to set
    a_bar = spr.coo_matrix((ones, (page_one, page_two)))  # Convert to SciPy COO Matrix
    a_bar.setdiag(1)  # Make all nodes adjacent to themselves

    a_dense = a_bar.todense()  # Convert to Dense to  easily split into test/train

    # Re-create two adjacency matrices for training/testing
    a_bar = a_dense[:split_t, :split_t]
    a_bar_test = a_dense[split_t:split_v, split_t:split_v]

    print(a_bar.shape)
    print(a_bar_test.shape)

    # Convert back to COO Matrix
    a_bar = spr.coo_matrix(a_bar)
    a_bar_test = spr.coo_matrix(a_bar_test)

    # Normalize
    a_bar = normalize_adjacency_matrix(a_bar=a_bar)
    a_bar_test = normalize_adjacency_matrix(a_bar=a_bar_test)

    # Convert to Sparse Tensor
    a_bar = coo_matrix_to_sparse_tensor(a_bar)
    a_bar_test = coo_matrix_to_sparse_tensor(a_bar_test)

    print(a_bar.shape)
    print(a_bar_test.shape)

    return train_feats, train_labels, a_bar, test_feats, test_labels, a_bar_test, val_feats, val_labels


def ensure_valid_split(train, test, val):
    if train+test+val == 1.0 and train==val:
        return True
    else:
        print("Train Split + Validation Split + Test Split must equal 1.0.")
        print("Validation Split and Test Split must currently also be equal")
        print("to support multiplication by by th Adjacency Matrix")
        print("Please ensure values for these variables sum to 1.0 and")
        print("that the Test Split and Validation Split are equal")
        exit(1)


def main():
    print("Tensorflow version:", tf.__version__)
    print("Numpy version:", np.__version__)
    file_path = r"C:\Users\johnv\Documents\University\COMP3710\Pattern Flow Project\facebook.npz"

    # Variables
    plot_tsne = False

    epochs = 100
    learning_rate = 0.05

    train_split = 0.80
    validate_split = 0.10
    test_split = 0.10

    ensure_valid_split(test_split, train_split, validate_split)

    # Load in Data
    data = np.load(file_path)
    # There are 22 470 Pages
    # Each with 128 features
    # Each falls into 1 of 4 categories
    # There are 342 004 Edges between the pages

    # test_split = 0.2
    train_feats, train_labels, a_bar, \
        test_feats, test_labels, a_bar_test, \
            val_feats, val_labels = parse_data(data,
                                               train_split,
                                               validate_split)

    # ================== REAL MODEL ========================
    print("=============== Building Model ===============")
    # Construct Model
    my_model = myGraphModel.makeMyModel(a_bar, a_bar_test, train_feats)

    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = op.Adam(learning_rate=learning_rate)
    my_model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    # ================== RUN MODEL ========================
    # Train Model
    my_model.fit(train_feats,
                 train_labels,
                 epochs=epochs,
                 batch_size=22470,
                 shuffle=True,
                 validation_data=(val_feats, val_labels)
                 )

    print(my_model.summary())

    # Evaluate Model
    my_model.evaluate(test_feats,
                      test_labels,
                      batch_size=22470)

    # Plot TSNE
    if plot_tsne:
        generate_tsne_plot(test_labels, test_feats)


if __name__ == '__main__':
    main()

