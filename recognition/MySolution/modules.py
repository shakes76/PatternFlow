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


def get_node_indices(graph, ids):
    node_ids = np.asarray(ids)
    flat_node_ids = node_ids.reshape(-1)
    flat_node_indices = graph.node_ids_to_ilocs(flat_node_ids)
    node_indices = flat_node_indices.reshape(1, len(node_ids))
    return node_indices


class Modules:
    def __init__(self, data):
        self.target_encoding = None
        self.test_gen = None
        self.val_gen = None
        self.gcn_model = None
        self.generator = None
        self.model = None
        self.train_gen = None
        data_group = self.get_training_data(data.get_target())
        self.build_model(data)
        self.data_group = data_group

    def get_training_data(self, target, train_size=200):
        train_data, test_data = train_test_split(target, train_size=train_size)
        val_data, test_data = train_test_split(test_data, train_size=train_size)
        self.target_encoding = LabelBinarizer()
        train_targets = self.target_encoding.fit_transform(train_data["page_type"])
        val_targets = self.target_encoding.fit_transform(val_data["page_type"])
        test_targets = self.target_encoding.fit_transform(test_data["page_type"])
        return train_data, val_data, test_data, train_targets, val_targets, test_targets
    
    def build_model(self, data):
        self.generator = FullBatchNodeGenerator(data.get_graph())
        corrupted_generator = CorruptedGenerator(self.generator)
        train_gen = corrupted_generator.flow(data.get_graph().nodes())
        self.gcn_model = GCN(
            layer_sizes=[16, 16],
            activations=["relu", "relu"],
            generator=self.generator,
            dropout=0.5
        )
        deep_graph = DeepGraphInfomax(self.gcn_model, corrupted_generator)
        x_in, x_out = deep_graph.in_out_tensors()
        model = Model(inputs=x_in, outputs=x_out)
        model.compile(
            loss=tf.nn.sigmoid_cross_entropy_with_logits,
            optimizer=optimizers.Adam(lr=1e-3)
        )
        self.model = model
        self.train_gen = train_gen

    def model_retrain(self, split_data):
        train_data, val_data, test_data, train_targets, val_targets, test_targets =\
            split_data
        self.train_gen = self.generator.flow(train_data.index, train_targets)
        self.test_gen = self.generator.flow(test_data.index, test_targets)
        self.val_gen = self.generator.flow(val_data.index, val_targets)
        x_in, x_out = self.gcn_model.in_out_tensors()
        predictions = layers.Dense(
            units=train_targets.shape[1],
            activation="softmax"
        )(x_out)
        model = Model(inputs=x_in, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(lr=0.01),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )
        return model

    def get_model(self):
        return self.model

    def get_train_gen(self):
        return self.train_gen

    def get_data_group(self):
        return self.data_group

    def get_gen(self):
        return self.train_gen, self.test_gen, self.val_gen

    def get_target_encoding(self):
        return self.target_encoding

    def get_generator(self):
        return self.generator
