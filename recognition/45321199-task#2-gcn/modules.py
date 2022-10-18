from dataset import DataLoader
import numpy as np
import tensorflow as tf

from tensorflow.keras import activations, initializers, regularizers, layers, models, optimizers

# Default Model Parameters
CHANNELS        = 16 
DROPOUT         = 0.3
LEARNING_RATE   = 1e-2 
REG_RATE        = 2.5e-4 
EPOCHS          = 40

class GCN_Model:
    def __init__(self, channels=CHANNELS, 
                dropout=DROPOUT, 
                learning_rate=LEARNING_RATE, 
                reg_rate=REG_RATE, 
                epochs=EPOCHS):

        self.channels = channels
        self.dropout = dropout 
        self.learning_rate = learning_rate
        self.reg_rate = reg_rate
        self.epochs = epochs
        self.model = None
        
        # get data
        data_loader = DataLoader()
        self.data = data_loader.load_data()

    def create(self):

        self.model.summary()

    def compile(self):
        self.model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate), 
                loss='categorical_crossentropy', 
                metrics = ['acc'])

class GCN_Layer(layers.Layer):
    def __init__(self, 
        activation, 
        channels=CHANNELS, 
        kernel_initialiser = 'glorot_uniform',
        kernel_regulariser = None):

        super().__init__() 
        self.channels = channels
        self.activation = activations.get(activation) 
        self.kernel_initialiser = initializers.get(kernel_initialiser)
        self.kernel_regulariser = regularizers.get(kernel_regulariser)
        

    def build(self, input_shape): 
        assert len(input_shape)>= 2

        self.w = self.add_weight(
                            shape=(input_shape[0][-1], self.channels), 
                            initializer=self.kernel_initialiser,
                            name="kernel",
                            regularizer=self.kernel_regulariser)

    def call(self, inputs):
        x, a = inputs

        output = np.dot(x, self.w)
        output = np.dot(a, output)

        return self.activation(output)