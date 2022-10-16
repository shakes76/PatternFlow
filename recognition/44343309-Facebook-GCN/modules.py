import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, regularizers, preprocessing, Model
from tensorflow.keras.layers import Layer
import matplotlib.pyplot as plt
import numpy as np

class GCN(Model):
  def __init__(self):
      super(GCN, self).__init__()
      
  def __call__(self, inputs, training=False):
        edges = inputs[1]
        self.model.add(GCNLayer(input_dim=128, output_dim=64, activation="relu", kernel_regularizer=l2(0.01))(inputs))
        self.model.add(layers.Dropout(0.3))
        self.model.add(GCNLayer(input_dim=64, output_dim=16, activation="relu")([self.model, edges]))
        self.model.add(layers.Dropout(0.3))
        self.model.add(GCNLayer(input_dim=16, output_dim=4, activation="softmax")([self.model, edges])) 

        return self.model

class GCNLayer(Layer):
  def __init__(self, inDim, outDim, activation=None, bias=False):
        super(GCNLayer, self).__init__()
        self.inDim = inDim
        self.outDim = outDim
        self.activation = activation
        self.weight = self.add_weight(shape=(self.input_dim, self.output_dim), initializer="glorot_uniform")
        self.built = True

    def __call__(self, inputs):
        features = inputs[0]
        edges = inputs[1]
        featWeights = tf.matmul(features, self.weight)
        out = tf.matmul(edges, featWeights)

        if self.activation is not None:
            out = self.activation(out)

        return out