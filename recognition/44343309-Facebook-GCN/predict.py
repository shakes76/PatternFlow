from modules import *
from dataset import *
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
#from sklearn.metrics import classification_report

class PredictionFromModel:
    def __init__(self, dataPath):
        self.dataset = DataProcess(dataPath)

        (self.features, 
        self.labels, 
        self.adjacency, 
        self.trainMask, 
        self.validaMask, 
        self.testMask, 
        self.trainLabels, 
        self.validaLabels, 
        self.testLabels, 
        self.target, 
        self.numNodes, 
        self.numFeatures) = self.dataset.getData()

        self.classes = len(np.unique(self.target))

    def generateModel(self):
        self.model = GCN(self.numNodes, self.numFeatures, self.classes)
        self.model.load_weights("training/cp.ckpt")
        return self.model
    
    def predictResults(self):
        return self.model.predict([self.features, self.adjacency], batch_size=self.numNodes)

    def plotTSNE(self):
        #report = classification_report(testMask, np.argmax(predictions, axis=1))
        gcnOutputs = [layer.output for layer in self.model.layers]
        activationModel = tf.keras.Model(inputs=self.model.input, outputs=gcnOutputs)
        activations = activationModel.predict([self.features, self.adjacency], batch_size=self.numNodes)
        tsne = TSNE(n_components=2).fit_transform(activations[3])

        # plot the tnse plot
        colourMap = np.argmax(self.labels, axis=1)
        plt.figure(figsize=(15, 15))
        for classes in range(self.classes):
            indices = np.where(colourMap == classes)
            indices = indices[0]
            plt.scatter(tsne[indices,0], tsne[indices, 1], label=classes)
        plt.legend()
        plt.grid()
        plt.show()


