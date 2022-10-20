import networkx as nx
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
import numpy as np
import stellargraph as sg
from tensorflow import keras


def get_file():
    """
    The get_file() function gets the facebook dataset from the provided
    website and extracted.

    Returns: the data directory as data_dir

    """

    zip_file = keras.utils.get_file(
        fname="Facebook_Data",
        origin="https://snap.stanford.edu/data/facebook_large.zip",
        extract=True,
    )
    return os.path.join(os.path.dirname(zip_file), "facebook_large")


def get_edges(data_dir):
    """
    The get_edges() function takes a data directory and read the csv file
    within.

    Args:
        data_dir: the path to the edge file

    Returns: The data read with panda.

    """
    edges = pd.read_csv(
        os.path.join(data_dir, "musae_facebook_edges.csv"),
        header=None,
        sep=",",
        names=["target", "source"],
        skiprows=1,
    )
    print("Edges shape:", edges.shape)
    return edges


def get_features(data_dir):
    """
    The get_features() function takes a data directory and read the json
    file within.

    Args:
        data_dir: the path to the features file

    Returns: The data read with json_load

    """
    feature_path = os.path.join(data_dir, "musae_facebook_features.json")
    with open(feature_path) as json_data:
        features = json.load(json_data)
    return features


def get_feature_matrix(features):
    """
    The get_feature_matrix() function use the given features to create a
    feature matrix. Initialise an empty matrix of the size of feature
    classes x number of features. Then label the matrix with ones for given
    features for processing

    Args:
        features: The table of features to be used

    Returns: the features in a matrix of ones and zeros

    """
    max_feature = np.max(
        [v for v_list in features.values() for v in v_list]
    )
    features_matrix = np.zeros(
        shape=(len(list(features.keys())),
               max_feature + 1)
    )
    i = 0
    for k, vs in features.items():
        for v in vs:
            features_matrix[i, v] = 1
        i += 1
    print("Feature matrix shape:", features_matrix.shape)
    return features_matrix


def get_node_features(features, feature_matrix):
    """
    The get_node_features() function insert the feature matrix into a pandas
    data frame using the features' keys as index

    Args:
        features: The features to get keys for
        feature_matrix: The matrix to be converted to pd.DataFrame

    Returns: the panda data frame for of the matrix

    """
    return pd.DataFrame(
        feature_matrix,
        index=features.keys(),
    )


def get_target(data_dir, features):
    """
    The get_target() function takes a data directory and read the csv file
    within.

    Args:
        features: the features of the dataset
        data_dir: the path to the target file

    Returns: The data read with panda.

    """
    target = pd.read_csv(
        os.path.join(data_dir, "musae_facebook_target.csv"),
        header=None,
        sep=",",
        names=["id", "facebook_id", "page_name", "page_type"],
        skiprows=1,
    )
    print("Target shape:", target.shape)
    target.index = target.id.astype(str)
    return target.loc[features.keys(), :]


def generate_graph(edges, target, samples):
    """
    The generate_graph() reates a graph that shows the connections between
    each node with each node colored by class.

    Args:
        edges: The edges to be used and connected
        target: The classes of each node
        samples: The number of nodes to display (reduce loading time)

    """
    class_values = sorted(target["page_type"].unique())
    target_values = sorted(target["id"].unique())
    class_idx = {name: ID for ID, name in enumerate(class_values)}
    target_idx = {name: idx for idx, name in enumerate(target_values)}
    target["id"] = target["id"].apply(lambda name: target_idx[name])
    edges["source"] = edges["source"].apply(lambda name: target_idx[name])
    edges["target"] = edges["target"].apply(lambda name: target_idx[name])
    target["page_type"] = target["page_type"].apply(lambda value: class_idx[value])
    plt.figure(figsize=(10, 10))
    facebook_graph = nx.from_pandas_edgelist(edges.sample(n=samples))
    subjects = list(target[target["id"].isin(list(facebook_graph.nodes))]["page_type"])
    nx.draw_spring(facebook_graph, node_size=10, node_color=subjects)
    plt.show()


class Dataset:
    """
    Class Dataset initialise the facebook dataset and generate a set of
    arrays and tables useful for training the model

    """
    def __init__(self):
        """
        Get the edges, features and target data from an online zip file
        and produce a set of useful data values.

        """
        data_dir = get_file()
        self.edges = get_edges(data_dir)
        self.features = get_features(data_dir)
        self.feature_matrix = get_feature_matrix(self.features)
        self.node_features = get_node_features(self.features, self.feature_matrix)
        self.target = get_target(data_dir, self.features)
        self.graph = sg.StellarGraph(self.node_features, self.edges.astype(str))

    def get_visualised_graph(self, samples=3000):
        """
        The get_visualised_graph() function calls the generate_graph()
        function to generate a graph of samples defined in the samples value

        Args:
            samples: The number of samples to display

        """
        generate_graph(self.edges, self.target, samples)

    def get_edges(self):
        """
        The get_edges() function returns the edges of this class

        Returns: the edges of this class

        """
        return self.edges

    def get_features(self):
        """
        The get_features() function returns the features of this class

        Returns: the features of this class

        """
        return self.features

    def get_feature_matrix(self):
        """
        The get_feature_matrix() function returns the feature matrix of
        this class

        Returns: the feature matrix of this class

        """
        return self.feature_matrix

    def get_node_features(self):
        """
        The get_node_features() function returns the node features of this
        class

        Returns: the node feature of this class

        """
        return self.node_features

    def get_target(self):
        """
        The get_target() function returns the target of this class

        Returns: the target of this class

        """
        return self.target

    def get_graph(self):
        """
        The get_graph function returns the stellargraph of this class

        Returns: the stellargraph of this class

        """
        return self.graph
