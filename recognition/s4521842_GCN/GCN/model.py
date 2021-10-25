import tensorflow as tf

from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, initializers, regularizers

class GraphConvolutionLayer(Layer):
    def __init__(self, input_dim, output_dim,
                 activation='relu', 
                 bias=True, 
                 kernel_initializer="glorot_uniform",
                 bias_initializer = 'zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 **kwargs):
        super(GraphConvolutionLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activations.get(activation)
        self.bias = bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer) 
        
        self.w = self.add_weight(name='kernal', 
                                 shape=(self.input_dim, self.output_dim), 
                                 initializer=self.kernel_initializer, 
                                 regularizer=self.kernel_regularizer)
        if self.bias:
            self.b = self.add_weight(name='bias',
                                     shape=(self.output_dim, ), 
                                     initializer=self.bias_initializer, 
                                     regularizer=self.bias_regularizer)
        else:
            self.b = None
            
        self.build = True

    
    def call(self, inputs):
        features = inputs[0]
        edges = inputs[1]
        support = tf.matmul(features, self.w) 
        output = tf.matmul(edges, support)

        if self.bias:
            output += self.b
            
        return self.activation(output)


class GCN(Model):
    def __in__(self):
        super(GCN, self).__init__(input_dim, output_dim)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.graph_conv_1 = GraphConvolutionLayer(input_dim, 16,
                    activation=tf.keras.activations.relu,
                    kernel_regularizer=tf.keras.regularizers.l2(0.01))

        self.graph_conv_2 = GraphConvolutionLayer(16, output_dim)

    def call(self, inputs, training=False):

        H = self.graph_conv_1(inputs)
        Y = self.graph_conv_2([H, inputs[1]])

        return Y
