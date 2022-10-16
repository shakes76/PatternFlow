import numpy as np
import scipy.sparse as sp
from keras.utils import np_utils
from sklearn.preprocessing import LabelBinarizer, normalize

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
    numTrain = int(self.numNodes*0.8)
    numTestValid = int(self.numNodes*0.1)
    trainLabels = self.target[:numTrain]
    testLabels = self.target[numTrain:numTrain + numTestValid]
    validaLabels = self.target[numTrain + numTestValid:numTrain + 2 * numTestValid]
    
    trainMask = np.zeros(self.numNodes, dtype=np.bool)
    testMask = np.zeros(self.numNodes, dtype=np.bool)
    validaMask = np.zeros(self.numNodes, dtype=np.bool)

    trainLabels[:numTrain] = True
    testLabels[numTrain:numTrain + numTestValid] = True
    validaLabels[numTrain + numTestValid:numTrain + 2 * numTestValid] = True

    return trainLabels, testLabels, validaLabels, trainMask, testMask, validaMask

  def processing(self):
    one_hot_labels = np_utils.to_categorical(self.target)
    self.features = normalize(self.features)

    adjMat = sp.coo_matrix(
            (np.ones(self.edges.shape[0]), (self.edges[:, 0], self.edges[:, 1])),
            shape=(self.target.shape[0], self.target.shape[0]),
            dtype=np.float64
        )

    normalAdj = normaliseAdjacency(adjMat)

    trainLabels, testLabels, validaLabels, trainMask, testMask, validaMask = self.splitData()
    trainLabels = LabelBinarizer().fit_transform(trainLabels)
    testLabels = LabelBinarizer().fit_transform(testLabels)
    validaLabels = LabelBinarizer().fit_transform(validaLabels)
    
    return self.features, one_hot_labels, normalAdj, trainMask, validaMask, testMask, trainLabels, validaLabels, testLabels


  def getData(self):
    return self.dataset

#misc functions for testing
def showFiles(data):
  print(data.files)
  return data.files

def organiseData(data):
  edges = data['edges']
  features = data['features']
  target = data['target']
  return edges, features, target

#def main():
 # facebook = np.load("facebook.npz")
 #dataset = DataLoadAndProcess()
 # datatat = dataset.getData()

#if __name__ == '__main__':
    #main()