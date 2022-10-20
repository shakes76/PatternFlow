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

# build a tnse plot
gcnOutputs = [layer.output for layer in model.layers]
activationModel = tf.keras.Model(inputs=model.input, outputs=gcnOutputs)
activations = activationModel.predict([features, adjacency], batch_size=numNodes)
tnseGraph = TSNE(n_components=2).fit_transform(activations[3])

# plot the tnse plot
colours = np.argmax(target, axis=1)
plt.figure(figsize=(15, 15))
for klass in range(classes):
    indices = np.where(colours == klass)
    indices = indices[0]
    plt.scatter(tnseGraph[indices,0], tnseGraph[indices, 1], label=klass)
plt.legend()
plt.grid()
plt.show()
