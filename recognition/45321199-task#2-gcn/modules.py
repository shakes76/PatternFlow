"""
    This file contains the class "GCN_Model" & "GCN_Layer", which handles creating the GCN itself.
"""

__author__ = "Lachlan Comino"


from dataset import DataLoader
import tensorflow as tf

from tensorflow.keras import backend, activations, initializers, regularizers, layers, Model, optimizers, losses

# Default Model Parameters
CHANNELS        = 256
DROPOUT         = 0.05
LEARNING_RATE   = 0.0001
REG_RATE        = 0.000005   # regularisation rate
KERNAL_REGULARIZER = regularizers.l2(REG_RATE)


class GCN_Model(Model):
    """
        Class to represent entire graph convolutional network.
    """
    def __init__(self, channels=CHANNELS, dropout=DROPOUT, learning_rate=LEARNING_RATE, kernal_regularizer=KERNAL_REGULARIZER):
        """ Creates the GCN as well as initialising the DataLoader
            @Param:
                channels: amount of output channels
                dropout: rate of dropout
                learning_rate: rate of learning
                kernal_regularizer: regularisation applier function
        """
        super(GCN_Model, self).__init__()
        self.channels = channels
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.kernal_regularizer = kernal_regularizer
        self.model = None
        
        # get data
        data_loader = DataLoader()
        self.data = data_loader.load_data()

        # create&compile model
        self.create()
        self.compile()

    def create(self):
        """
            Creates the inputs, middle layer and output layer of the GCN.
        """
        # Input layers
        x_input = layers.Input((self.data['len_features'],), dtype=tf.float64)
        node_input = layers.Input((self.data['len_vertices'],), dtype=tf.float64, sparse=True)
        
        # Middle layers
        dropout_L0 = layers.Dropout(self.dropout)(x_input)
        gcn_L0 = GCN_Layer(activation=activations.relu, 
                            channels=self.channels, 
                            kernel_regulariser=self.kernal_regularizer)([dropout_L0, node_input])

        dropout_L1 = layers.Dropout(self.dropout)(gcn_L0)
        gcn_L1 = GCN_Layer(activation=activations.relu, 
                             channels=self.channels // 2,
                             kernel_regulariser=self.kernal_regularizer)([dropout_L1, node_input])
        # Output layer
        dropout_L2 = layers.Dropout(self.dropout)(gcn_L1)
        gcn_L2 = GCN_Layer(activation=activations.softmax, 
                            channels=self.data['len_label_types'])([dropout_L2, node_input])
        
        self.model = Model(inputs=[x_input, node_input], outputs=gcn_L2)
        self.model.summary()


    def compile(self):
        """
            Compiles the GCN. Uses Adam optomiser and categorical cross entropy
        """
        if self.model is not None:
            self.model.compile(
                    optimizer=optimizers.Adam(learning_rate=self.learning_rate), 
                    loss=losses.CategoricalCrossentropy(), 
                    metrics = ['acc'])


class GCN_Layer(layers.Layer):
    """
        Class to implement custom layer functionality for the GCN.
    """
    def __init__(self, 
                    activation=activations.relu, 
                    channels=CHANNELS, 
                    kernel_initialiser='glorot_uniform',
                    kernel_regulariser=None, 
                    **kwargs):
        """ Initialises the layer.
            @Param:
                activation: type of activation function
                channels: amount of channels in output
                kernal_initialiser: weight initialiser
                kernal_regularizer: regularisation applier function
        """

        super(GCN_Layer, self).__init__(**kwargs)
        self.channels = channels
        self.activation = activations.get(activation) 
        self.kernel_initialiser = initializers.get(kernel_initialiser)
        self.kernel_regulariser = regularizers.get(kernel_regulariser)
        

    def build(self, input_shape):
        self.w = self.add_weight(
                            shape=(input_shape[0][-1], self.channels), 
                            initializer=self.kernel_initialiser,
                            name="kernel",
                            regularizer=self.kernel_regulariser)

    def call(self, inputs):
        x, a = inputs

        output = backend.dot(x, self.w)
        output = backend.dot(a, output)

        return self.activation(output)
    
    def get_config(self):
        """
            Required for loading custom layers. Helps tf
            interpret what is being loading from the .h5 file.
        """
        config = super(GCN_Layer, self).get_config()
        config.update({"channels": self.channels})
        return config