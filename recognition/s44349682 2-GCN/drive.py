import tensorflow as tf
from tensorflow.keras import callbacks, optimizers

import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from model import GCN

def main():
    # Settings / Hyperparams
    dataset = "facebook.npz"
    epochs = 200
    learning_rate = 0.01

    # Read the dataset
    fb_data = np.load(dataset)

    ## Extract data
    # Number of Nodes
    nodes = len(fb_data['target'])
    classes = np.unique(fb_data['target'])

    # Feature list
    X = fb_data['features']
    # Normalise features of each node
    X /= X.sum(1).reshape(-1, 1)
    X = tf.convert_to_tensor(X)

    # Edge list
    edges = fb_data['edges']
    # Convert edge list to sparse edge coordinate matrix
    A = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(nodes, nodes), dtype=np.float32)

    # One-hot encode labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(fb_data['target'])

    # Preprocess adjacency matrix by adding identity and dot with degree matrix
    # Giving D^(-1/2) . (A + I) . D^(-1/2)
    # Necessary for correctly calculating adjacent feature values
    AStar = A + sp.eye(A.shape[0])
    D = sp.diags(np.power(np.array(AStar.sum(1)), -0.5).flatten())
    AStar = D.dot(AStar).dot(D).tocoo()
    # Convert to a TF Sparse Array
    indices = np.mat([AStar.row, AStar.col]).transpose()
    AStar = tf.SparseTensor(indices, AStar.data, AStar.shape)

    # Establish Graph Input to GCN
    graph = [X, AStar]

    ## Create Masks for train/val/test sets, using 0.2/0.2/0.6 split
    train_mask = np.zeros(nodes, dtype=np.bool)
    val_mask = np.zeros(nodes, dtype=np.bool)
    test_mask = np.zeros(nodes, dtype=np.bool)

    train_range = range(int(nodes/5))
    val_range = range(int(nodes/5), int(2*nodes/5))
    test_range = range(int(2*nodes/5), nodes)

    train_mask[train_range] = True
    val_mask[train_range] = True
    test_mask[train_range] = True

    validation_data = (graph, labels, val_mask)

    ## Running the Model
    # Initialise
    gcn = GCN()

    # Compile model using categorical crossentropy, Adam optimizer
    # and record accuracy metric while training
    gcn.compile(loss='categorical_crossentropy',
                optimizer=optimizers.Adam(learning_rate=learning_rate),
                metrics=['accuracy'],
                run_eagerly=True)

    # Early Stop callback to prevent overfitting of model
    early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                            restore_best_weights=True)

    # Fit the model to the graph using the training set mask
    history = gcn.fit(graph,
                    labels,
                    sample_weight=train_mask,
                    epochs=epochs,
                    batch_size=nodes,
                    validation_data=validation_data,
                    shuffle=False,
                    callbacks=[early_stop]
                    )

    ## Displaying Results
    # Show plots of loss and accuracy during fitting over epochs
    plot1 = plt.figure(1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim([0, 0.5])
    plt.legend(loc='lower right')

    plot2 = plt.figure(2)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Predict classifications of the entire graph
    y_pred = gcn.predict(graph, batch_size = nodes)

    # Print classification report of prediction precision
    report = classification_report(np.argmax(labels,axis=1), np.argmax(y_pred,axis=1))
    print(report)

    # Generate 2D TSNE graph of nodes
    tsne = TSNE(n_components=2).fit_transform(y_pred)
    color_map = np.argmax(labels, axis=1)
    plt.figure(3, figsize=(10,10))
    for i in range(len(classes)):
        indices = np.where(color_map==i)
        indices = indices[0]
        plt.scatter(tsne[indices,0], tsne[indices, 1], label=i)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()