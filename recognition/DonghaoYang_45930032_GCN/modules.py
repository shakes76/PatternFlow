"""
modules.py containing the source code of the components of my GCN model.
"""

# All needed library for creating a GCN model.
import tensorflow as tf
from keras import layers
from keras import activations
from keras import initializers
from keras import regularizers
from keras import Model
from keras import Input
from keras import backend


class GraphConvolution(layers.Layer):
    """
    This class contains the basic graph convolution layer
    """

    def __init__(self, output_dimension, activation_function=None, use_bias=True,
                 kernel_initializer="glorot_uniform", kernel_regularizer=None, bias_initializer='zeros',
                 bias_regularizer=None):
        self.output_dimension = output_dimension
        self.activation_function = activations.get(activation_function)
        self.use_bias = use_bias
        # initializer function defines the function "glorot_uniform" to set initialised weights
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # load the function for regularization
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        super(GraphConvolution, self).__init__()

    def build(self, input_shape):
        """
        add weights to the layer
        """
        features_shape = input_shape[0]
        input_dimension = features_shape[1]
        if not hasattr(self, 'weight'):
            self.weight = self.add_weight(name='weight', shape=(input_dimension, self.output_dimension),
                                          initializer=self.kernel_initializer,
                                          regularizer=self.kernel_regularizer,
                                          trainable=True)
        if not hasattr(self, 'bias'):
            if self.use_bias:
                self.bias = self.add_weight(name='bias', shape=(self.output_dimension,),
                                            initializer=self.bias_initializer,
                                            regularizer=self.bias_regularizer,
                                            trainable=True)
        self.built = True
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs, *args, **kwargs):
        """
        The calculation of core equation of symmetric normalisation GCN
        """
        features, edges = inputs
        output = tf.matmul(features, self.weight)
        output = backend.dot(edges, output)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation_function is not None:
            output = self.activation_function(output)
        return output

    def get_config(self):
        """
        Override the "get_config()"
        """
        config = super().get_config()
        return config


def GCN(features_matrix):
    """
    This function define a multi layers GCN
    """
    features_number, nodes_number = features_matrix.shape[1], features_matrix.shape[0]
    # define input
    input_feature = Input((features_number,))
    input_nodes = Input((nodes_number,))
    # layer1
    gcn_layer1 = GraphConvolution(64, activation_function=activations.relu)([input_feature, input_nodes])
    dropout_layer1 = layers.Dropout(0.5)(gcn_layer1)
    # layer2
    gcn_layer2 = GraphConvolution(32, activation_function=activations.relu)([dropout_layer1, input_nodes])
    dropout_layer2 = layers.Dropout(0.5)(gcn_layer2)
    # layer3
    gcn_layer3 = GraphConvolution(16, activation_function=activations.relu)([dropout_layer2, input_nodes])
    dropout_layer3 = layers.Dropout(0.5)(gcn_layer3)
    # layer4
    gcn_layer4 = GraphConvolution(4, activation_function=activations.softmax)([dropout_layer3, input_nodes])
    model = Model(inputs=[input_feature, input_nodes], outputs=gcn_layer4)
    return model
