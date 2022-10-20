# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:24:13 2022

@author: Danie
"""

import tensorflow as tf 
import pathlib
import tarfile
import numpy as np
import matplotlib as plt
import tensorflow.keras as kr
import glob as gb

from StableDiffusionModel import *
from CustomLayers import *
from AutoEncoder import *


##
##  TODO: MAKE BETTER DOCSTRINGS
##


def downloadOASIS( destinationFolder = "./DataSets" ):
    """
        Downloads the OASIS Brain MRI dataset.
    """
    dataURL = "https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download"
    dataDirectory = tf.keras.utils.get_file(origin=dataURL,fname='oa-sis' ,untar=True)
    dataDirectory = pathlib.Path(dataDirectory)
    #file = tarfile.open("C:\Users\Danie\.keras\datasets")
    #file.extractall('./DataSets')
    
    

def loadTrainingData (path, dataType = np.float32):
    """
        Downloads the OASIS Brain MRI dataset.
    """
    imageList = []
    for filename in gb.glob(path+'/*.png'): 
        img=plt.image.imread (filename)
        imageList.append(img)

    print('Raw training data shape:',np.array(imageList).shape)
    trainSet = np.array(imageList, dtype=dataType)
    return trainSet

def processTrainingData(rawData, newImSize = 128):
    """
        Resizes data and organises new axis for input into tensorflow
    """

    trainData = rawData
    
    # Rearranging data for call to tf.image.resize
    trainData = tf.transpose(trainData, [1, 2, 0])

    # Resizing data
    trainData = tf.image.resize(trainData, (newImSize, newImSize))
    
    # Rearranging data back to (B, W, H, C) format
    trainData = tf.transpose(trainData, [2, 0, 1])
    
    # Adding new axis
    trainData = trainData[:, :, :, np.newaxis]

    # Converting to tensor
    trainData = tf.convert_to_tensor(trainData)
    
    # Unflipping Data (its flipped in the resize step for some reason)
    trainData = (1-trainData)   
    
    return trainData


def autoEncoderBuildCompile(inputSize = 128, 
                            normLayers = True, 
                            activation = kr.activations.relu, 
                            loadWeights = False, 
                            loadWeightsPath = ""):
    """
    Builds and compiles the AutoEncoder defined in AutoEncoder.py, 
    and readies it for training.
    """
    
    autoEnc = AutoEncoder(inputSize, inputSize/4, normLayers = True, activation = activation)
    
    if (loadWeights) :
        autoEnc.load_weights(loadWeightsPath)
    
    autoEnc.compile(optimizer = kr.optimizers.Adam(), loss="mse")
    return autoEnc
    

def trainAutoEncoder(dataSet, batch_size = 32, epochs = 600, saveModel = True, saveLoc = "./CheckPoints/autoEncChecpoint{Epoch}"):
    """
        Trains the input autoencoder. Optionally saves checkpoints (enabled by default)
        
        Returns the history object storing the loss observed across the training
    """
    if (saveModel):
        save_model = tf.keras.callbacks.ModelCheckpoint(
            filepath=saveLoc,
            save_weights_only=True,
            monitor='val_accuracy',)
        callbacks = [save_model]
    else :
        callbacks = []
        
        history = autoEnc.fit(dataSet, 
                              dataSet, 
                              epochs = epochs, 
                              batch_size = batch_size, 
                              callbacks=callbacks)
        return history




def plotAutoEncoderExamples(autoEncoder, dataSet, inputDim=128, imagesShown = 3, fontSize = 10):
    """
    Plots images against their latent space and reconstruction from the autoencoder.   

    Parameters
    ----------
    autoEncoder : AutoEncoder
        AutoEncoder model to plot results from
        
    dataSet : Tensor
        Image data set autoencoder was trained on
        
    inputDim : TYPE, optional
        Size of the image, e.g. 64 for a 64x64 image
        
    imagesShown : TYPE, optional
        Number of Images plotted
        
    fontSize : TYPE, optional
        font size of titles in plots
    ----------

    """
    
    fig, axs = plt.pyplot.subplots(imagesShown,3)

    newInput = kr.Input((inputDim, inputDim, 1))
    encoder = kr.models.Model(newInput, autoEnc.encoder(newInput))

    for row in range(0, imagesShown) :
      testImage = dataSet[row]
      testImage = testImage[np.newaxis, :, :, :]
      latentImage = encoder(testImage)
      reconstructedImage = autoEncoder(testImage)
      axs[row, 0].imshow(tf.squeeze(testImage), cmap='Greys')
      axs[row, 0].set_title("Original Image {row} ".format(row=(row+1)), fontsize=fontSize)
      axs[row, 1].imshow(tf.squeeze(latentImage), cmap='Greys')
      axs[row, 1].set_title("Latent Space {row} ".format(row=(row+1)), fontsize=fontSize)
      axs[row, 2].imshow(tf.squeeze(reconstructedImage), cmap='Greys')
      axs[row, 2].set_title("Reconstructed Image {row} ".format(row=(row+1)), fontsize=fontSize)

    for row in range(0,imagesShown) :
      for col in range(0,3) :
        axs[row, col].get_xaxis().set_visible(False)
        axs[row, col].get_yaxis().set_visible(False)
    fig.tight_layout()


if __name__ == "__main__":
    downloadOASIS()
    trainDataRaw = loadTrainingData('./DataSets/keras_png_slices_train')
    trainData = processTrainingData(trainDataRaw)
    
    autoEnc = autoEncoderBuildCompile(loadWeights = True, loadWeightsPath = "./CheckPoints/FinalModel")
    history = trainAutoEncoder(trainData)
    
    plotAutoEncoderExamples(autoEnc, trainData, imagesShown = 5, fontSize = 8)
    
    
    