import tensorflow as tf

from tensorflow.keras.layers import Layer, Dropout
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import activations, initializers, regularizers
import tensorflow.keras.backend as K

# GCN Layer, performs operation to collect adjacent feature lists
# and generate classification prediction based on output of matrices.
# Updates bias to learn and improve predictions
class GraphConvolutionLayer(Layer):

    # Constructor for GCN Layer
    def __init__(self, input_dim, output_dim,
                    activation=None,
                    bias=True,
                    kernel_initializer="glorot_uniform",
                    bias_initializer='zeros',
                    kernel_regularizer=None,
                    bias_regularizer=None):
        super(GraphConvolutionLayer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.use_bias = bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        # Create weight kernel to propogate
        self.weight = self.add_weight(name='kernel',
                                    shape=(self.input_dim, self.output_dim),
                                    initializer=self.kernel_initializer,
                                    regularizer=self.kernel_regularizer)

        # If bias is enabled, create a bias weighting
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(self.output_dim,),
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer)
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):
        
        # Collect features and adjacency matrix
        features = inputs[0]
        A = inputs[1]
        
        # Matrix operations to calculate output based on adjacent features
        support = K.dot(features, self.weight)
        output = K.dot(A, support)

        # Add bias if enabled
        if self.use_bias:
            output += self.bias

        # Use activation function to generate output
        if self.activation is not None:
            output = self.activation(output)

        return output