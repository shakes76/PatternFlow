from dataset import *
from modules import *  
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models, regularizers, preprocessing, Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

optimizer = Adam(learning_rate=0.001)

dataset = DataLoadAndProcess()

(features, labels, adjacency,
 trainMask, validaMask, testMask,
 trainLabels, validaLabels, testLabels, target, numNodes, numFeatures) = dataset.getData()

classes = len(np.unique(target))

model = GCN(numNodes, numFeatures, classes)

checkpointPath = "training/cp.ckpt"
checkpointDir = os.path.dirname(checkpointPath)

#create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpointPath, save_weights_only=True, verbose=1, save_freq=10)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', weighted_metrics=['acc'])

model.summary()

validation_data = ([features, adjacency], labels, validaMask)

model.fit([features, adjacency],
          labels,
          sample_weight=trainMask,
          epochs=50,
          batch_size=numNodes,
          validation_data=validation_data,
          shuffle=False,
          callbacks=[cp_callback]
 )