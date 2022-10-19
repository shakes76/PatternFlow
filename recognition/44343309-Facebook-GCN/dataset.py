import numpy as np
import scipy.sparse as sp
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer, normalize

def loadData(data):
    dataset = np.load(data)
    edges = dataset["edges"]
    features = dataset["features"]
    target = dataset["target"]

    return edges, features, target

data = loadData("facebook.npz")

def normaliseAdjacency(adjacency):
    adjacency += sp.eye(adjacency.shape[0])
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()

class DataProcess:
  def __init__(self):
    self.edges, self.features, self.target = loadData("facebook.npz")
    self.numNodes = self.features.shape[0]
    self.numFeatures = self.features.shape[1]

    self.dataset = self.processing()
    
  def splitData(self):
    """
    split is a cool 80:10:10
    """
    trainSplit = int((self.numNodes) * 0.8)
    trainSplit = range(trainSplit)
    validaSplit = range(trainSplit, trainSplit + int((self.numNodes) * 0.1))
    testSplit = range(trainSplit + int((self.numNodes) * 0.1), self.numNodes)

    trainMask = np.zeros(self.numNodes, dtype=np.bool)
    validaMask = np.zeros(self.numNodes, dtype=np.bool)
    testMask = np.zeros(self.numNodes, dtype=np.bool)
    trainMask[trainSplit] = True
    validaMask[validaSplit] = True
    testMask[testSplit] = True

    # split labels into training, validation, test set
    trainLabels = self.target[trainSplit]
    validaLabels = self.target[validaSplit]
    testLabels = self.target[testSplit]
    return trainLabels, testLabels, validaLabels, trainMask, testMask, validaMask

    return trainLabels, testLabels, validaLabels, trainMask, testMask, validaMask

  def processing(self):
    one_hot_labels = np_utils.to_categorical(self.target)
    self.features = normalize(self.features)

    adjMat = sp.coo_matrix(
            (np.ones(self.edges.shape[0]), 
            (self.edges[:, 0], self.edges[:, 1])),
            shape=(self.target.shape[0], 
            self.target.shape[0]),
            dtype=np.float32
        )

    normalAdj = normaliseAdjacency(adjMat)

    trainLabels, testLabels, validaLabels, trainMask, testMask, validaMask = self.splitData()
    trainLabels = LabelBinarizer().fit_transform(trainLabels)
    testLabels = LabelBinarizer().fit_transform(testLabels)
    validaLabels = LabelBinarizer().fit_transform(validaLabels)
    
    return self.features, one_hot_labels, normalAdj, trainMask, validaMask, testMask, trainLabels, validaLabels, testLabels, self.target, self.numNodes, self.numFeatures


  def getData(self):
    return self.dataset
