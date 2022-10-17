import tensorflow as tf
import numpy as np
import stellargraph as sg
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from stellargraph.mapper import FullBatchNodeGenerator
from tensorflow import keras
from keras import layers, Input, Model, optimizers, losses
from keras.layers import Dropout, Dense
from stellargraph.layer.gcn import GraphConvolution, GatherIndices, GCN
from keras.callbacks import EarlyStopping
import dataset


def get_training_data(target, train_size=200):
    train_data, test_data = train_test_split(target, train_size=train_size)
    val_data, test_data = train_test_split(test_data, train_size=train_size)
    target_encoding = LabelBinarizer()
    train_targets = target_encoding.fit_transform(train_data["page_type"])
    val_targets = target_encoding.fit_transform(val_data["page_type"])
    test_targets = target_encoding.fit_transform(test_data["page_type"])
    return train_data, val_data, test_data, train_targets, val_targets, test_targets


def get_node_indices(graph, ids):
    node_ids = np.asarray(ids)
    flat_node_ids = node_ids.reshape(-1)
    flat_node_indices = graph.node_ids_to_ilocs(flat_node_ids)
    node_indices = flat_node_indices.reshape(1, len(node_ids))
    return node_indices


def build_model(data, split_data):
    train_data, val_data, test_data, train_targets, val_targets, test_targets = \
        split_data
    generator = FullBatchNodeGenerator(data.get_graph(), method="gcn")
    train_gen = generator.flow(train_data.index, train_targets)
    gcn_model = GCN(
        layer_sizes=[16, 16],
        activations=["relu", "relu"],
        generator=generator,
        dropout=0.5
    )
    x_inp, x_out = gcn_model.in_out_tensors()
    predictions = layers.Dense(
        units=train_targets.shape[1],
        activation="softmax"
    )(x_out)
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.01),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )
    val_gen = generator.flow(val_data.index, val_targets)
    return model, val_gen, train_gen


class Modules:
    def __init__(self, data):
        data_group = get_training_data(data.get_target())
        self.model, self.val_gen, self.train_gen = build_model(data, data_group)

    def get_model(self):
        return self.model

    def get_val_gen(self):
        return self.val_gen

    def get_train_gen(self):
        return self.train_gen


