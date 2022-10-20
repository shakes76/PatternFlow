import tensorflow as tf
from tensorflow.keras import Model, activations, initializers
from tensorflow.keras.layers import Layer, Input, Dropout

def GCN(numNodes, numFeatures, numClasses, channelA=64, channelB=32, channelC=16, channelD=8, dropout=0.1):
  features = Input(shape=(numFeatures))
  nodes = Input((numNodes), sparse=True)

  dp1 = Dropout(dropout)(features)
  layer1 = GCNLayer(channelA, activation='relu')([dp1, nodes])

  dp2 = Dropout(dropout)(layer1)
  layer2 = GCNLayer(channelB, activation='relu')([dp2, nodes])

  dp3 = Dropout(dropout)(layer2)
  layer3 = GCNLayer(channelC, activation='relu')([dp3, nodes])

  dp4 = Dropout(dropout)(layer3)
  layer4 = GCNLayer(channelD, activation='relu')([dp4, nodes])

  dp5 = Dropout(dropout)(layer4)
  layer5 = GCNLayer(numClasses, activation='softmax')([dp5, nodes])

  model = Model(inputs=[features, nodes], outputs=layer5)
    
  return model
class GCNLayer(Layer):
  def __init__(self, channels, activation=None, kernel_initializer='glorot_uniform', 
              bias_initializer='zeros'):
    super(GCNLayer, self).__init__(
        trainable=True)
    self.channels = channels
    self.activation = activations.get(activation)
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

  def build(self, input):
    inDim = input[0][-1]
    self.kernel = self.add_weight(name="kernel",
        shape=(inDim, self.channels),
        initializer=self.kernel_initializer
      )
    self.built = True

  def call(self, input):
    features, edges = input
    output = tf.keras.backend.dot(features, self.kernel)
    output = tf.keras.backend.dot(edges, output)
    output = self.activation(output)

    return output