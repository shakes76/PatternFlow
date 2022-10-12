import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers


# Get the data from the provided link
def get_file():
    zip_file = keras.utils.get_file(
        fname="Facebook_Data",
        origin="https://snap.stanford.edu/data/facebook_large.zip",
        extract=True,
    )
    data_dir = os.path.join(os.path.dirname(zip_file), "facebook_large")
    return data_dir


def get_edges(data_dir):
    edges = pd.read_csv(
        os.path.join(data_dir, "musae_facebook_edges.csv"),
        header=None,
        delimiter=',',
        names=["target", "source"],
    )
    print("Edges shape:", edges.shape)
    return edges


def get_target(data_dir):
    target = pd.read_csv(
        os.path.join(data_dir, "musae_facebook_target.csv"),
        header=None,
        delimiter=',',
        names=["id", "facebook_id", "page_name", "page_type"]
    )
    print("target shape:", target.shape)
    return target


def handle_dataset():
    data_dir = get_file()
    edges = get_edges(data_dir)
    target = get_target(data_dir)
    print(edges.sample(frac=1).head())
    print(target.sample(frac=1).T)
    print(target.page_type.value_counts())





