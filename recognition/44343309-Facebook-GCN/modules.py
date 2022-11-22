"""
Author: Remington Greenhill-Brown
SN: 44343309
"""
import tensorflow as tf
from tensorflow.keras import Model, activations, initializers
from tensorflow.keras.layers import Layer, Input, Dropout

"""
GCN(): creates a tensorflow model for data to be passed through. passed through 4 relu layers before a final softmax layer. 
params: number of nodes in dataset, number of classes/labels in dataset, 4 channels of halving size, and a dropout value
returns: a model ready to be fit by tensorflow (in tensorflow Model format)
"""
def GCN(numNodes, numFeatures, numClasses, channelA=64, channelB=32, channelC=16, channelD=8):
  features = Input(shape=(numFeatures))
  nodes = Input((numNodes), sparse=True)

  # layers that data is passed through. each has a decreasing channel size and most use relu as the activation
  layer1 = GCNLayer(channelA, activation='relu')([features, nodes])
  layer2 = GCNLayer(channelB, activation='relu')([layer1, nodes])
  layer3 = GCNLayer(channelC, activation='relu')([layer2, nodes])
  layer4 = GCNLayer(channelD, activation='relu')([layer3, nodes])
  # final layer, uses softmax to map output to zero or one
  layer5 = GCNLayer(numClasses, activation='softmax')([layer4, nodes])

  # creates model from keras Model class
  model = Model(inputs=[features, nodes], outputs=layer5)
    
  return model

"""
GCNLayer(Layer): class to create each layer for use in model creation
"""
class GCNLayer(Layer):
  """
  __init__(channels, activation, kernel initialiser, bias initialiser): initialises data for class, calls super on the Layer class in keras.
  """
  def __init__(self, channels, activation=None, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros'):
    super(GCNLayer, self).__init__(
        trainable=True)
    self.channels = channels
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

  """
  build(input): takes an input and creates weights
  returns: true
  """
  def build(self, input):
    inDim = input[0][-1]
    # adds weights
    self.kernel = self.add_weight(name="kernel",
        shape=(inDim, self.channels),
        initializer=self.kernel_initializer
      )
    self.built = True

  """
  call(input): applies layers to the input
  returns: an output with layers applied
  """
  def call(self, input):
    features, edges = input
    output = tf.keras.backend.dot(features, self.kernel)
    output = tf.keras.backend.dot(edges, output)
    output = self.activation(output)

    return output