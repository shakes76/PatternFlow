"""
Author: Remington Greenhill-Brown
SN: 44343309
"""
import numpy as np
import scipy as sp
import tensorflow as tf
from sklearn.preprocessing import normalize

"""
loadData(): function to load data
params: data - dataset file in npz format
returns: the edges, features, and targets (classes/labels) of a chosen dataset
"""
def loadData(data):
    dataset = np.load(data)
    edges = dataset["edges"]
    features = dataset["features"]
    target = dataset["target"]

    return edges, features, target

"""
DataProcess: a class to load and process the data input into a usable format
"""
class DataProcess:
  def __init__(self, path):
    self.edges, self.features, self.target = loadData(path)
    self.numNodes = self.features.shape[0]
    self.numFeatures = self.features.shape[1]

    self.dataset = self.processing()
    
  """
  splitData(): does what it says, splits the data into train, validation, and test sets
  split is a cool 80:10:10
  returns: train labels, test labels, validation labels, train mask, test mask, validation mask
  """
  def splitData(self):
    # 80% train
    numTrain = int(self.numNodes*0.8)
    # 10% each for test and validation
    numTestValid = int(self.numNodes*0.1)

    # splits the train labels according the specified split
    trainLabels = self.target[:numTrain]
    testLabels = self.target[numTrain:numTrain + numTestValid]
    validaLabels = self.target[numTrain + numTestValid:numTrain + 2 * numTestValid]
    
    # set up train/test/validation masks as np array of zeroes
    trainMask = testMask = validaMask = np.zeros(self.numNodes, dtype=np.bool)

    # change zeroes of mask to ones for each relevant mask
    trainMask[:numTrain] = True
    testMask[numTrain:numTrain + numTestValid] = True
    validaMask[numTrain + numTestValid:numTrain + 2 * numTestValid] = True

    return trainLabels, testLabels, validaLabels, trainMask, testMask, validaMask

  """
  processing(): creates adjacency matrix, transforms labels/classes to one hot encoding for use in model
  returns: everything needed for model creation: all features, one hot encoded labels, a normalised adjacency matrix, training/validation/testing masks, 
           training/validation/testing labels, the actual labels themselves, the number of nodes in the dataset, and the number of features in the dataset
  """
  def processing(self):
    trainLabels, testLabels, validaLabels, trainMask, testMask, validaMask = self.splitData()
    labelsOneHot = tf.keras.utils.to_categorical(self.target, 4)
    adjMat = sp.sparse.coo_matrix(
            (np.ones(self.edges.shape[0]), 
            (self.edges[:, 0], self.edges[:, 1])),
            shape=(self.target.shape[0], self.target.shape[0]),
            dtype=np.float32)
    normalAdj = normalize(adjMat, norm='l1', axis=1)

    # changes label data to one hot encoding instead of 'random' numbers
    trainLabels = tf.keras.utils.to_categorical(trainLabels, 4)
    testLabels = tf.keras.utils.to_categorical(testLabels, 4)
    validaLabels = tf.keras.utils.to_categorical(validaLabels, 4)

    return self.features, labelsOneHot, normalAdj, trainMask, validaMask, testMask, trainLabels, validaLabels, testLabels, self.target, self.numNodes, self.numFeatures

  """
  getData(): simple function to return the processed data
  returns: the processed data
  """
  def getData(self):
    return self.dataset
