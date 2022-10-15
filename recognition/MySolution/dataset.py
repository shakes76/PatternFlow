import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy as sp
import requests
from pandas import DataFrame
from scipy import sparse
from tensorflow import keras
from keras import layers


def separate_dataset(dataset, debug=False):
    edges = dataset["edges"]
    features = dataset["features"]
    target = dataset["target"]
    if debug:
        print("Edges shape is: " + str(edges.shape))
        print("Features shape is: " + str(features.shape))
        print("Target shape is: " + str(target.shape))
    return edges, features, target


def handle_dataset():
    directory = "C:/Users/Sam Wang/Desktop/COMP3710_Report/facebook.npz"
    dataset = np.load(directory)
    return separate_dataset(dataset, debug=True)

