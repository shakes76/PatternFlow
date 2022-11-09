import tensorflow as tf

from tensorflow.keras.layers import Layer, Dropout
from tensorflow.keras import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, softmax


class GraphConvolutionLayer(Layer):
    """
        Graph convolution layer
    """

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
        self.activation = activation
        self.bias = bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer

        # create kernal to the layer
        self.w = self.add_weight(name='kernal',
                                 shape=(self.input_dim, self.output_dim),
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer)

        # if use bias, create bias to the layer
        if self.bias:
            self.b = self.add_weight(name='bias',
                                     shape=(self.output_dim,),
                                     initializer=self.bias_initializer,
                                     regularizer=self.bias_regularizer)
        else:
            self.b = None

        self.built = True

    def call(self, inputs):
        features = inputs[0]
        edges = inputs[1]
        support = tf.matmul(features, self.w)
        output = tf.matmul(edges, support)

        if self.bias:
            output += self.b

        if self.activation is not None:
            output = self.activation(output)

        return output


class GCN(Model):
    def __init__(self):
        super(GCN, self).__init__()
        # add gcn layer
        self.graph_conv_1 = GraphConvolutionLayer(input_dim=128,
                                                  output_dim=64,
                                                  activation=relu,
                                                  kernel_regularizer=l2(0.01))
        self.graph_conv_2 = GraphConvolutionLayer(input_dim=64, output_dim=16, activation=relu)
        self.graph_conv_3 = GraphConvolutionLayer(input_dim=16, output_dim=4, activation=softmax)

    def call(self, inputs, training=False):
        edges = inputs[1]
        H = self.graph_conv_1(inputs)
        H = Dropout(0.3)(H)
        H = self.graph_conv_2([H, edges])
        H = Dropout(0.3)(H)
        Y = self.graph_conv_3([H, edges])

        return Y
