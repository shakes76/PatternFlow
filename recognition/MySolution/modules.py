import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from stellargraph.layer import DeepGraphInfomax
from stellargraph.mapper import CorruptedGenerator, FullBatchNodeGenerator
from keras import layers, Model, optimizers
from stellargraph.layer.gcn import GCN


class Modules:
    """
    The Modules class stores variables of the created GCN models which
    can be called

    """
    def __init__(self, data):
        """
        takes the data variable from the dataset class to build a GCN model

        Args:
            data: the dataset class to be used for building the model

        """
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
        """
        The get_training_data() function takes the target dataset and separate
        the data into training, validation and testing sets.

        Args:
            target: the dataset to be split into training, validation and testing
            train_size: the size of the training set

        Returns: the split data and the binary encoded version of the classes

        """
        train_data, test_data = train_test_split(target, train_size=train_size)
        val_data, test_data = train_test_split(test_data, train_size=train_size)
        self.target_encoding = LabelBinarizer()
        train_targets = self.target_encoding.fit_transform(train_data["page_type"])
        val_targets = self.target_encoding.fit_transform(val_data["page_type"])
        test_targets = self.target_encoding.fit_transform(test_data["page_type"])
        return train_data, val_data, test_data, train_targets, val_targets, test_targets
    
    def build_model(self, data):
        """
        The build_model() function takes the dataset class to build a GCN
        model. The generator shuffles node features and regular node features
        to train the model to differentiate the real and fake. The GCN model
        will use 2 hidden layers of 32 under relu.

        Args:
            data: The dataset class to be used for generating the model

        """
        self.generator = FullBatchNodeGenerator(data.get_graph())
        corrupted_generator = CorruptedGenerator(self.generator)
        train_gen = corrupted_generator.flow(data.get_graph().nodes())
        self.gcn_model = GCN(
            layer_sizes=[32, 32],
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
        """
        The model_retrain() function takes the split data to retrain the
        GCN model created in the build_model() function.

        Args:
            split_data: A total of 6 datasets returned by the get_training_data()
                        function

        Returns: the new model for the final training

        """
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
        """
        The get_model() function gets the pretraining model of this class

        Returns: the pretraining model of this class

        """
        return self.model

    def get_train_gen(self):
        """
        The get_train_gen() function gets the train generator values of this class

        Returns: the train generator flow object

        """
        return self.train_gen

    def get_data_group(self):
        """
        The get_data_group() function gets the group of train, val and test data

        Returns: the train, val and test data

        """
        return self.data_group

    def get_gen(self):
        """
        The get_gen() function return the train, test and val generators together

        Returns: the train, test and val generator flow objects

        """
        return self.train_gen, self.test_gen, self.val_gen

    def get_target_encoding(self):
        """
        The get_target_encoding() function returns the encoded target values

        Returns: the encoded target values

        """
        return self.target_encoding

    def get_generator(self):
        """
        The get_generator() function returns the generator used by the model

        Returns: the generator used by the model

        """
        return self.generator
