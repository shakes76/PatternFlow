from dataset import DataLoader
import tensorflow as tf

from tensorflow.keras import backend, activations, initializers, regularizers, layers, Model, optimizers

# Default Model Parameters
CHANNELS        = 32
DROPOUT         = 0.4
LEARNING_RATE   = 0.01


class GCN_Model:
    def __init__(self, channels=CHANNELS, dropout=DROPOUT):
        self.channels = channels
        self.dropout = dropout 
        self.model = None
        
        # get data
        data_loader = DataLoader()
        self.data = data_loader.load_data()

        # make tensors
        self.tf_features = tf.convert_to_tensor(self.data['features'])
        self.tf_graph = tf.convert_to_tensor(self.data["graph_adj_mat"])

    def create(self):
        x_input = layers.Input((self.data["len_features"],), dtype=tf.float64)
        node_input = layers.Input((self.data["len_vertices"],), dtype=tf.float64, sparse=True)

        dropout_L0 = layers.Dropout(self.dropout)(x_input)
        gcn_L0 = GCN_Layer(activations.relu, self.channels)([dropout_L0, node_input])

        dropout_L1 = layers.Dropout(self.dropout)(gcn_L0)
        gcn_L1 = GCN_Layer(activations.relu, self.data["len_label_types"])([dropout_L1, node_input])
        
        self.model = Model(inputs = [x_input, node_input], outputs=gcn_L1)
        self.model.summary()

    def compile(self):
        if self.model is None:
            self.model.compile(
                    optimizer=optimizers.Adam(learning_rate=self.learning_rate), 
                    loss='categorical_crossentropy', 
                    metrics = ['acc'])


class GCN_Layer(layers.Layer):
    def __init__(self, 
        activation, 
        channels=CHANNELS, 
        kernel_initialiser='glorot_uniform',
        kernel_regulariser=None):

        super().__init__() 
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