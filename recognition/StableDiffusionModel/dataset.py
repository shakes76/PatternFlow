# -*- coding: utf-8 -*-
"""
@author: Daniel Ju Lian Wong
"""

import matplotlib as plt
import numpy as np
import tensorflow as tf
import pathlib
import glob as gb


def downloadOASIS( destinationFolder = "./DataSets" ):
    """
    Downloads the OASIS Brain MRI dataset.
    """
    dataURL = "https://cloudstor.aarnet.edu.au/plus/s/tByzSZzvvVh0hZA/download"
    dataDirectory = tf.keras.utils.get_file(origin=dataURL,fname='oa-sis' ,untar=True)
    dataDirectory = pathlib.Path(dataDirectory)


def loadTrainingData (path, dataType = np.float32):
    """
    Loads the OASIS Brain MRI dataset
    """
    imageList = []
    for filename in gb.glob(path+'/*.png'): 
        img=plt.image.imread (filename)
        imageList.append(img)
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

