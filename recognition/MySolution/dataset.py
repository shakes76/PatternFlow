import pandas as pd
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from scipy import sparse


def transform_features_to_sparse(table):
    """
    Transforms a given table of 2 rows and n columns to a sparse matrix.
    Take the table and add a new row called 'weight'. Convert the table
    to a list and then to a sparse matrix with scipy import

    Args:
        table: The table to convert to a sparse matrix

    Returns: the converted sparse matrix

    """
    table["weight"] = 1
    table = table.values.tolist()
    index_1 = [row[0] for row in table]
    index_2 = [row[1] for row in table]
    values = [row[2] for row in table]
    count_1, count_2 = max(index_1) + 1, max(index_2) + 1
    sp_m = sparse.csr_matrix(sparse.coo_matrix((
        values, (index_1, index_2)),
        shape=(count_1, count_2),
        dtype=np.float32)
    )
    return sp_m


def generate_graph(edges, target):
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


def normalize_adjacency(raw_edges, targets):
    """
    Normalise the adjacency of nodes and return teh features transformed.
    Take the table of edges and reverse the edges before stacking both
    tables together. This ensures that every edge is included.
    Then use the networkx import to find the node angles to be used to
    find the do product of the edges

    Args:
        targets: The table of targets
        raw_edges: The table of edges

    Returns: the transformed edges

    """
    raw_edges_t = pd.DataFrame()
    raw_edges_t["id_1"] = raw_edges["id_2"]
    raw_edges_t["id_2"] = raw_edges["id_1"]
    raw_edges = pd.concat([raw_edges, raw_edges_t])
    edges = raw_edges.values.tolist()
    # generate_graph(raw_edges, targets)
    graph = nx.from_edgelist(edges)
    size = range(len(graph.nodes()))
    degrees = [1.0 / graph.degree(node) for node in graph.nodes()]
    transformed_edges = transform_features_to_sparse(raw_edges)
    degrees = sparse.csr_matrix(sparse.coo_matrix(
        (degrees, (size, size)),
        shape=transformed_edges.shape,
        dtype=np.float32)
    )
    transformed_edges = transformed_edges.dot(degrees)
    return transformed_edges


def class_classifier(given_class):
    if given_class == "politician":
        numeric_class = 0
    elif given_class == "company":
        numeric_class = 1
    elif given_class == "government":
        numeric_class = 2
    elif given_class == "tvshow":
        numeric_class = 3
    else:
        raise Exception("unknown class")
    return numeric_class


def handle_dataset():
    target_location = "C:/Users/Sam Wang/Desktop/COMP3710_Report/musae_facebook_target.csv"
    features_location = "C:/Users/Sam Wang/Desktop/COMP3710_Report/musae_facebook_features.csv"
    edges_location = "C:/Users/Sam Wang/Desktop/COMP3710_Report/musae_facebook_edges.csv"
    targets = pd.read_csv(target_location)
    features = pd.read_csv(features_location)
    edges = pd.read_csv(edges_location)
    target = targets["page_type"].values.tolist()
    classes = np.array([class_classifier(t) for t in target])
    processed_edges = normalize_adjacency(edges, targets)
    processed_features = transform_features_to_sparse(features)
    processed_features_tilde = processed_edges.dot(processed_features)
    return

