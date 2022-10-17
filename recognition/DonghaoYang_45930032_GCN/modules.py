"""
modules.py containing the source code of the components of my GCN model.
"""

# All needed library for creating a GCN model.
import tensorflow as tf
from keras import layers
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints


class GraphConvolution(layers.Layer):
    """
    This class contains the basic graph convolution layer
    """
    def __init__(self, output_dimension, activation_function=None, use_bias=True,
                 kernel_initializer="glorot_uniform", kernel_regularizer=None, bias_initializer='zeros',
                 bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None):
        self.output_dimension = output_dimension
        self.activation_function = activations.get(activation_function)
        self.use_bias = use_bias
        # initializer function defines the function "glorot_uniform" to set initialised weights
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        # load the function for regularization
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        # constraint is the function to add constraints to weights
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)
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
                                          constraint=self.kernel_constraint,
                                          regularizer=self.kernel_regularizer,
                                          trainable=True)
        if not hasattr(self, 'bias'):
            if self.use_bias:
                self.bias = self.add_weight(name='bias', shape=(self.output_dimension,),
                                            initializer=self.bias_initializer,
                                            constraint= self.bias_constraint,
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
        output = tf.matmul(edges, output)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        if self.activation is not None:
            output = self.activation(output)
        return output
