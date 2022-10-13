import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, regularizers, preprocessing
import matplotlib.pyplot as plt
import numpy as np

class GCN(tensorflow.keras.Model):
  def __init__(self):
    super(GCN, self).__init__()
    #add layers here


class GCNLayer(tensorflow.keras.layers.Layer):
  def __init__(self, inDim, outDim):
    super(GCNLayer, self).__init__()
    self.inputDim = inDim
    self.outputDim = outDim
    