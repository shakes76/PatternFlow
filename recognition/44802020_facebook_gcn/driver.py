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


def plot_tsne(labels, feats):
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


def parse_data(data, test_split):
    # Adjacency Matrix
    # Split EdgeList into two tensors
    page_one = data['edges'][:, 0]
    page_two = data['edges'][:, 1]
    # Features
    feats = tf.convert_to_tensor(data['features'])
    # Labels
    labels = tf.convert_to_tensor(data['target'])

    # Convert split percentage into integer
    split = int(round(labels.shape[0] * test_split))
    train_labels, test_labels = labels[:split], labels[split:]
    train_feats, test_feats = feats[:split], feats[split:]

    # Convert EdgeList to Sparse Adjacency Matrix
    ones = tf.ones_like(page_one)  # Create Ones Matrix to set
    a_bar = spr.coo_matrix((ones, (page_one, page_two)))  # Convert to SciPy COO Matrix
    a_bar.setdiag(1)  # Make all nodes adjacent to themselves

    a_dense = a_bar.todense()  # Convert to Dense to  easily split into test/train

    # Re-create two adjacency matrices for training/testing
    a_bar = a_dense[:split, :split]
    a_bar_test = a_dense[split:, split:]

    # Convert back to COO Matrix
    a_bar = spr.coo_matrix(a_bar)
    a_bar_test = spr.coo_matrix(a_bar_test)

    # Normalize
    a_bar = normalize_adjacency_matrix(a_bar=a_bar)
    a_bar_test = normalize_adjacency_matrix(a_bar=a_bar_test)

    # Convert to Sparse Tensor
    a_bar = coo_matrix_to_sparse_tensor(a_bar)
    a_bar_test = coo_matrix_to_sparse_tensor(a_bar_test)

    return train_feats, train_labels, a_bar, test_feats, test_labels, a_bar_test


def main():
    print("Tensorflow version:", tf.__version__)
    print("Numpy version:", np.__version__)

    file_path = r"C:\Users\johnv\Documents\University\COMP3710\Pattern Flow Project\facebook.npz"

    # Load in Data
    data = np.load(file_path)

    # There are 22 470 Pages
    # Each with 128 features
    # Each falls into 1 of 4 categories
    # There are 342 004 Edges between the pages

    test_split = 0.2
    train_feats, train_labels, a_bar, test_feats, test_labels, a_bar_test = parse_data(data, test_split)

    # ================== REAL MODEL ========================
    print("=============== Building Model ===============")
    # Construct Model
    my_model = myGraphModel.makeMyModel(a_bar, a_bar_test, train_feats)

    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    opt = op.Adam(learning_rate=0.05)
    my_model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

    # ================== RUN MODEL ========================
    # Train Model
    my_model.fit(train_feats,
                 train_labels,
                 epochs=50,
                 batch_size=22470, shuffle=False
                 )

    print(my_model.summary())

    # Evaluate Model
    my_model.evaluate(test_feats,
                      test_labels,
                      batch_size=22470)

    # Plot TSNE
    plot_tsne(test_labels, test_feats)


if __name__ == '__main__':
    main()

