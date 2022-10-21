# -*- coding: utf-8 -*-
"""
@author: Daniel Ju Lian Wong
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

from modules import *



def autoEncoderBuildCompile(inputSize = 128, 
                            normLayers = True, 
                            activation = kr.activations.relu, 
                            loadWeights = False, 
                            loadWeightsPath = "./Weights/FinalModel"):
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
    autoEncoder : AutoEncoder
        AutoEncoder model to plot results from
    dataSet : Tensor
        Image data set autoencoder was trained on
    inputDim : int, optional
        Size of the image, e.g. 64 for a 64x64 image
    imagesShown : int, optional
        Number of Images plotted
    fontSize : int, optional
        font size of titles in plots

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

def lossFunction(real, generated):
    """
    Implementation of mean squared error for use in the training function
    """
    loss = tf.math.reduce_mean((real - generated) ** 2)
    return loss

def trainOnBatch(batch):
    """
    Trains the stable diffusion model on the input batch
    """
    rng, tsrng = np.random.randint(0, 100000, size=(2,))
    timestep_values = createTimeStamp(tsrng, batch.shape[0])

    noised_image, noise = addNoise(rng, batch, timestep_values)
    with tf.GradientTape() as tape:
        prediction = unet(noised_image, timestep_values)
        
        lossValue = lossFunction(noise, prediction)
    
    gradients = tape.gradient(lossValue, unet.trainable_variables)
    opt.apply_gradients(zip(gradients, unet.trainable_variables))

    return lossValue

def plotHistory(history):
    """ plots the loss associated with the input training history """
    
    plt.pyplot.plot(history.history["loss"])

def checkPointManager(model, path):
  """
  Creates and returns a checkpoint manager for the input model. The checkpoint
  manager will write to the specified path
  """
  check = tf.train.Checkpoint(unet=model)
  checkManager = tf.train.CheckpointManager(check, path, max_to_keep=2)
  return check, checkManager



def trainLoop(epochs, encoder, losses):
    for e in range(1, epochs+1):
        bar = tf.keras.utils.Progbar(len(trainingData)-1)
        for i, batch in enumerate(iter(trainingData)):
            # Reducing the image to its latent representation
            latent = encoder(batch)
    
            loss = train_step(latent)
            losses.append(loss)
            bar.update(i, values=[("loss", loss)])
    
        avg = np.mean(losses)
        print(f"Mean loss {e}/{epochs}: {avg}")
        ckpt_manager.save(checkpoint_number=e)



if __name__ == "__main__":
    downloadOASIS()
    trainDataRaw = loadTrainingData('/content/keras_png_slices_data/keras_png_slices_train')
    trainData = processTrainingData(trainDataRaw)

    autoEnc = autoEncoderBuildCompile(loadWeights = True, loadWeightsPath = "/content/FinalModel")
    
    # Training the AutoEncoder
    history = trainAutoEncoder(trainData)
    plotHistory(history)

    # Plotting examples of autoEncoder reducing images to latent space
    plotAutoEncoderExamples(autoEnc, trainData, imagesShown = 3, fontSize = 8)

    # Initalising unet for diffusion model
    unet = Unet(channels=1, dim=64)
    
    # creating checkpoint manager
    ckpt, ckpt_manager= checkPointManager(model, path)
    
    # load from a previous checkpoint if it exists, else initialize the model from scratch
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        start_interation = int(ckpt_manager.latest_checkpoint.split("-")[-1])
        print("Restored from checkpoint {}".format(ckpt_manager.latest_checkpoint))
    else:
        print("Creating new checkpoint sequence")
    
    # Optimizer used in training
    opt = kr.optimizers.Adam(learning_rate=1e-4)
    
    # Batching and shuffling training data
    trainingData = tf.data.Dataset.from_tensor_slices(trainData).shuffle(1000).batch(64)
    
    # Training diffusion model
    losses = []
    trainLoop(epochs = 50, encoder = autoEnc.buildEncoder(), losses = losses)
    
    
    
    