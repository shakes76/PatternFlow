from modules import *
from dataset import *
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report

dataset = DataProcess()

(features, labels, adjacency,
trainMask, validaMask, testMask,
trainLabels, valLabels, testLabels, target, numNodes, numFeatures) = dataset.getData()

classes = len(np.unique(target))

model = GCN(numNodes, numFeatures, classes)

model.load_weights("training/cp.ckpt")

predictions = model.predict([features, adjacency], batch_size=numNodes)

report = classification_report(testMask, np.argmax(predictions, axis=1))