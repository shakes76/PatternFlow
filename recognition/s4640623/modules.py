import numpy as np
import tensorflow as tf
from tensorflow import keras
print(tf.__version__)

class GraphConvolutionLayer(tf.keras.Model):

  def __init__(self, input_features, output_features. **kwargs):
    super(GraphConvolutionLayer, self).__init__(**kwargs)
    self._input_features = input_features
    self._output_features = output_features
    
    self.optimiser = tf.train.AdamOptimizer(1e-4)

  # Calculates loss of model
  def _loss(self):
    #TODO
  
  # Calculates accuracy of model
  def _accuracy(self):
    #TODO
  
  # Builds model with graph convolution layers
  def _build(self):
    #TODO
  
  def predict(self):
    #TODO