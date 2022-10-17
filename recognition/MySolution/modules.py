import tensorflow as tf
import numpy as np
import stellargraph as sg
from scipy import sparse as sp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from stellargraph.layer import DeepGraphInfomax
from stellargraph.mapper import CorruptedGenerator, FullBatchNodeGenerator
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


class Modules:
    def __init__(self, data):
        self.model = None
        self.train_gen = None
        data_group = get_training_data(data.get_target())
        self.build_model(data, data_group)
        self.data_group = data_group

    def build_model(self, data, split_data):
        train_data, val_data, test_data, train_targets, val_targets, test_targets = \
            split_data
        generator = FullBatchNodeGenerator(data.get_graph())
        corrupted_generator = CorruptedGenerator(generator)
        train_gen = corrupted_generator.flow(data.get_graph().nodes())
        gcn_model = GCN(
            layer_sizes=[16, 16],
            activations=["relu", "relu"],
            generator=generator,
            dropout=0.5
        )
        deep_graph = DeepGraphInfomax(gcn_model, corrupted_generator)
        x_in, x_out = deep_graph.in_out_tensors()
        model = Model(inputs=x_in, outputs=x_out)
        model.compile(
            loss=tf.nn.sigmoid_cross_entropy_with_logits,
            optimizer=optimizers.Adam(lr=1e-3)
        )
        self.model = model
        self.train_gen = train_gen
    
    def get_model(self):
        return self.model

    def get_train_gen(self):
        return self.train_gen

    def get_data_group(self):
        return self.data_group


