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

class ModelTrainer:
    def __init__(self, checkpointPath):
        self.optimizer = Adam(learning_rate=0.001)
        self.dataset = DataProcess()

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

        self.model = GCN(self.numNodes, self.numFeatures, self.classes)
        self.validation_data = ([self.features, self.adjacency], self.labels, self.validaMask)

        self.checkpointDir = os.path.dirname(checkpointPath)
        #create checkpoint callback
        self.cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpointPath, save_weights_only=True, verbose=1, save_freq=10)

    def generateModel(self):
        self.model.compile(optimizer=self.optimizer, loss='categorical_crossentropy', weighted_metrics=['acc'])
        self.model.fit(
            [self.features, self.adjacency],
            self.labels,
            sample_weight=self.trainMask,
            epochs=50,
            batch_size=self.numNodes,
            validation_data=self.validation_data,
            shuffle=False,
            callbacks=[self.cp_callback]
        )
        return self.model

def main():
    model = ModelTrainer("training/cp.ckpt")
    model.generateModel()

if __name__ == '__main__':
    main()




