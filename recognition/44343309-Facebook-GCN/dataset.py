import numpy as np

def loadData():
  return np.load("C:/users/Remy/Downloads/facebook.npz")

class dataLoadAndProcess:
  def __init__(self):
    self.data = loadData()
    self.edges
    self.features
    self.target
    self.numNodes
    self.numFeatures

    self.dataset

  def allocateData(self):
    self.edges = self.data["edges"]
    self.features = self.data["features"]
    self.target = self.data["target"]

    self.numNodes = self.features.shape[0]
    self.numFeatures = self.features.shape[1]
    
  def splitData(self):
    """
    split is a cool 80:10:10
    """
    numTrain = len(self.numNodes)*0.8
    numTestValid = len(self.numNodes)*0.1
    trainLabels = self.target[:numTrain]
    testLabels = self.target[numTrain:numTrain + numTestValid]
    validaLabels = self.target[numTrain + numTestValid:numTrain + 2 * numTestValid]
    print(len(trainLabels))

  def getData(self):
    return self.dataset

def showFiles(data):
  print(data.files)
  return data.files

def organiseData(data):
  edges = data['edges']
  features = data['features']
  target = data['target']
  return edges, features, target



def main():
    data = loadData()
    showFiles(data)
    edges, features, target = organiseData(data)


if __name__ == '__main__':
    main()
