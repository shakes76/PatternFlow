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
        # A.H.W
        support = K.dot(features, self.weight)
        output = K.dot(A, support)

        # Add bias if enabled
        if self.use_bias:
            output += self.bias

        # Use activation function to generate output
        if self.activation is not None:
            output = self.activation(output)

        return output

# GCN Model, contains 2 Convolutional Layers, with dimensions 128->32->4
# Layer 1 uses relu activation while layer 2 uses softmax for accurate classification
# as each node only has a single label
class GCN(Model):
    def __init__(self):
        super(GCN, self).__init__()
        
        self.layer1 = GraphConvolutionLayer(input_dim=128,
                                            output_dim=32,
                                            activation=activations.relu,
                                            kernel_regularizer=l2(5e-4))
        
        self.layer2 = GraphConvolutionLayer(input_dim=32,
                                            output_dim=4,
                                            activation=activations.softmax)

    def call(self, inputs):
        # Extract Adjacency matrix to be input to each conv layer
        A = inputs[1]
        
        # Structure used from Kipf (2017)
        H = Dropout(0.5)(inputs[0])
        H = self.layer1([H, A])
        H = Dropout(0.5)(H)
        Y = self.layer2([H, A])

        return Y