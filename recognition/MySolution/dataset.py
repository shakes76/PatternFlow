import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy as sp
from tensorflow import keras
from keras import layers


# Get the data from the provided link
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
    data_dir = os.path.join(os.path.dirname(zip_file), "facebook_large")
    return data_dir


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
        # delimiter=',',
        sep=",",
        names=["target", "source"],
        skiprows=1,
    )
    print("Edges shape:", edges.shape)
    return edges


def get_target(data_dir):
    """
    The get_target() function takes a data directory and read the csv file
    within.

    Args:
        data_dir: the path to the target file

    Returns: The data read with panda.

    """
    target = pd.read_csv(
        os.path.join(data_dir, "musae_facebook_target.csv"),
        header=None,
        # delimiter=',',
        sep=",",
        names=["id", "facebook_id", "page_name", "page_type"],
        skiprows=1,
    )
    print("Target shape:", target.shape)
    return target


def sort_target_indices(edges, target):
    class_values = sorted(target["page_type"].unique())
    target_values = sorted(target["id"].unique())
    class_idx = {name: ID for ID, name in enumerate(class_values)}
    target_idx = {name: idx for idx, name in enumerate(target_values)}

    target["id"] = target["id"].apply(lambda name: target_idx[name])
    edges["source"] = edges["source"].apply(lambda name: target_idx[name])
    edges["target"] = edges["target"].apply(lambda name: target_idx[name])
    target["page_type"] = target["page_type"].apply(lambda value: class_idx[value])
    
    plt.figure(figsize=(10, 10))
    color = target["page_type"].tolist()
    facebook_graph = nx.from_pandas_edgelist(edges.sample(n=5000))
    subjects = list(target[target["id"].isin(list(facebook_graph.nodes))]["page_type"])
    nx.draw_spring(facebook_graph, node_size=15, node_color=subjects)
    plt.show()


def handle_dataset():
    data_dir = get_file()
    edges = get_edges(data_dir)
    edges.sample(frac=1).head()
    target = get_target(data_dir)
    print(target.sample(4).T)
    print(target.page_type.value_counts())
    sort_target_indices(edges, target)





