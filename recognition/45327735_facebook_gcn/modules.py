"""
The GCN model to be used to classify the Facebook Large Page-Page Network.

The model takes a preprocessed matrix of graph data (connections, features, weights) as an input
and predicts a label for each node.

Created on Fri Oct 07 12:48:34 2022

@author: Crispian Yeomans
"""

from keras import layers
from keras import Sequential
import tensorflow as tf
from dataset import Dataset

class GNN(tf.keras.Model):
    """
    Builds the GNN model using GraphConvLayers and vanilla Feed Forward Networks. Predicts the class of each node.
    """
    def __init__(self, sample_graph: Dataset, num_classes, hidden_nodes, aggregation_type="sum", combination_type="concat",
                 dropout_rate=0.2, normalise=True, *args, **kwargs,):
        super(GNN, self).__init__(*args, **kwargs)

        # Default values
        self.num_classes = num_classes
        self.hidden_nodes = hidden_nodes
        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.dropout_rate = dropout_rate
        self.normalise = normalise
        self.node_embeddings = None

        # The graph that will be used for training and predicting
        self.edges = sample_graph.get_edges()
        self.features = sample_graph.get_features()
        self.edge_weights = sample_graph.get_weights()

        # Build model architecture
        self._default_architecture()

    def _default_architecture(self):
        """
        Defines the architecture of the GNN using default values.
        """
        self._architecture(self.hidden_nodes, self.dropout_rate, self.aggregation_type, self.combination_type,
                           self.normalise, self.num_classes)

    def _architecture(self, hidden_nodes, dropout_rate, aggregation_type, combination_type, normalize, num_classes):
        """
        Defines the architecture of the GNN.

        - Feed Forward Networks are used to process input and output representations.
        - GraphConv layers are used to process graph data.
        - A n-node Dense layer (where n = number of classes) outputs the final prediction
        """
        # Create a process layer.
        self.preprocess = create_ffn(hidden_nodes, dropout_rate, name="preprocess")

        # Create the GraphConv layers.
        self.convs = []
        for i in range(0, 3):
            new_layer = GraphConvLayer(hidden_nodes, dropout_rate, aggregation_type,
                                    combination_type, normalize, name="graph_conv"+str(i))
            self.convs.append(new_layer)

        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_nodes, dropout_rate, name="postprocess")

        # Create a compute logits layer.
        self.predict_labels = layers.Dense(units=num_classes, name="predict_labels")

    def _predict_labels_for_indices(self, indices):
        """Returns network's predictions for the given node indices"""
        # Fetch node embeddings for the input indices
        node_embeddings = tf.gather(self.node_embeddings, indices)

        return self.predict_labels(node_embeddings)

    def call(self, input_node_indices):
        """
        Trains the network on the given graph, and outputs predictions for the nodes corresponding to the given indices.

        It also builds the model architecture, including residual single-skip connections between GraphConv layers.
        """
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.features)

        # Apply graph layers
        for i, conv in enumerate(self.convs):
            # Apply graph conv layer
            applied_layer = conv((x, self.edges, self.edge_weights))
            # Skip connection
            x = applied_layer + x

        # Postprocess final node representation
        self.node_embeddings = self.postprocess(x)

        # Predict labels
        return self._predict_labels_for_indices(input_node_indices)

"""
The following GraphConvLayer and create_ffn source code is from Keras. Comments by me.

Title: Graph Convolutional Network source code
Author: [Khalid Salama](https://www.linkedin.com/in/khalid-salama-24403144/)
Date created: 2021/05/30
Last modified: 2021/05/30
Source: https://keras.io/examples/graph/gnn_citations/
"""

def create_ffn(hidden_units, dropout_rate, name=None):
    """
    A vanilla Feedforward Neural Network that learns a 'filter' that is applied to the input.

    The network normalises the input batch, drops nodes (to mitigate memorising) and applies the filter.
    """
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu)) # the filter

    return Sequential(fnn_layers, name=name)


class GraphConvLayer(layers.Layer):
    """
    A GraphConvLayer that takes a (node_representations, edges, edge_weights) input and outputs a new node representation.

    Specifically, the layer aggregates neighbouring nodes data to produce a "message" for each node. These messages
    are combined with the old node representations to produce a new representation for each node.
    """
    def __init__(self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_representations, weights=None):
        """
        Produces an "outgoing message" for each node.

        Specifically, mode_representations are fed through a Feedforward Neural Network that learns weights for the
        node_representation (i.e., which features are most important) and produces a message.

        Messages will be aggregated by neighbouring nodes.
        """
        # node_representations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_representations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1) # applies constant weights (if any)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_representations):
        """
        Aggregates the neighbouring messages for each node.

        Each aggregation function - sum, mean, max - is permutation invariant.
        """
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        num_nodes = node_representations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        """
        Combines the aggregated messages and the old node representations to produce a new representation.

        Values are combined using a combination function: gru, concat, add. Combined values are then fed through a
        Feedforward Neural Network that has learnt weights used to process and output a new representation.
        """
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the new node representations (aka the node_embeddings).

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_representations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_representations shape is [num_edges, representation_dim].
        neighbour_representations = tf.gather(node_representations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_representations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_representations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_representations, aggregated_messages)