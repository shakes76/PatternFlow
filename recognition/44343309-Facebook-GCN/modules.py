import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, regularizers, preprocessing, Model, regularizers, activations, initializers, constraints, backend
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.layers import Layer, Input, Dropout
import matplotlib.pyplot as plt
import numpy as np


def GCN(numNodes, numFeatures, numClasses, channelA=32, channelB=8, dropout=0.1):
  features = Input(shape=(numFeatures))
  nodes = Input((numNodes), sparse=True)

  dp1 = Dropout(dropout)(features)
  layer1 = GCNLayer(channelA, activation='relu')([dp1, nodes])

  dp2 = Dropout(dropout)(layer1)
  layer2 = GCNLayer(channelB, activation='relu')([dp2, nodes])

  dp3 = Dropout(dropout)(layer2)
  layer3 = GCNLayer(numClasses, activation='softmax')([dp3, nodes])

  model = Model(inputs=[features, nodes], outputs=layer3)
    
  return model
class GCNLayer(Layer):
  def __init__(self, inDim, outDim, activation=None, bias=False):
    super(GCNLayer, self).__init__()
    self.inDim = inDim
    self.outDim = outDim
    self.activation = activation
    self.weight = self.add_weight(shape=(self.inDim, self.outDim), initializer="glorot_uniform")
    self.built = True

  def call(self, inputs):
    features = inputs[0]
    edges = inputs[1]
    featWeights = tf.matmul(features, self.weight)
    out = tf.matmul(edges, featWeights)

    if self.activation is not None:
      out = self.activation(out)

    return out