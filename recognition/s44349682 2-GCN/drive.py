import tensorflow as tf
from tensorflow.keras import callbacks, optimizers, regularizers, losses, backend
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import CategoricalCrossentropy

import numpy as np
import scipy.sparse as sp

from sklearn.preprocessing import LabelBinarizer

from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from model import *

# Settings / Hyperparams
dataset = "facebook.npz"
epochs = 50
learning_rate = 0.01

# Read the dataset
fb_data = np.load(dataset)

## Extract data
# Number of Nodes
nodes = len(fb_data['target'])
classes = np.unique(fb_data['target'])

# Feature list
X = fb_data['features']
# Normalise
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
AStar = A + sp.eye(A.shape[0])
D = sp.diags(np.power(np.array(AStar.sum(1)), -0.5).flatten())
AStar = D.dot(AStar).dot(D).tocoo()
indices = np.mat([AStar.row, AStar.col]).transpose()
AStar = tf.SparseTensor(indices, AStar.data, AStar.shape)

graph = [X, AStar]


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

GCN = GCN()

GCN.compile(loss='categorical_crossentropy',#loss=losses.SparseCategoricalCrossentropy(),
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            metrics=['accuracy'],
            run_eagerly=True)

early_stop = callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                        restore_best_weights=True)

history = GCN.fit(graph,
                labels,
                sample_weight=train_mask,
                epochs=epochs,
                batch_size=nodes,
                validation_data=validation_data,
                shuffle=False,
                callbacks=[early_stop]
                )



