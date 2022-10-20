# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 13:33:35 2022

@author: Danie
"""

import tensorflow.keras as kr
import tensorflow as tf
import matplotlib as plt
import numpy as np

from keras import layers
from CustomLayers import *


def sinusoidalTimeEmbedding(input):
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(1.0),
            tf.math.log(1000.0),
            32 // 2,   # latentDim//2
        )
    )
    embeddings = tf.concat(
        [tf.sin(2.0 * math.pi * frequencies * input), tf.cos(2.0 * math.pi * frequencies * input)], axis=3
    )
    return embeddings


class UNetBlock(kr.layers.Layer) :
    def __init__(self, outputSize, blockSize = 32, upBlock = False, upSampler = ConvUpsample, downSampler = ConvDownsample) :
        super().__init__()
        self.blockSize = blockSize
        self.outputSize = outputSize
        
        self.conv1 = kr.layers.Conv2D(outputSize, 3, padding = "same", activation = kr.activations.relu)
        self.bNorm1 = kr.layers.BatchNormalization()

        self.conv2 = kr.layers.Conv2D(outputSize, 3, padding = "same", activation = kr.activations.relu)
        self.bNorm2 = kr.layers.BatchNormalization()
        
        #self.embedder = kr.layers.Dense(blockSize)
        #self.embedderConv = kr.layers.Conv2D(outputSize, 3, padding = "same", activation = kr.activations.relu)
        
        # If upsampling occurs in block
        if (upBlock) :
            self.transformer = upSampler(outputSize)
        else :
            self.transformer = downSampler(outputSize)
    
    def call(self, input) :
        
        
        #print("ImageIS")
        #print(imageInput)

        #print("PRE EMBEDDING:", timeEmbedding)
        #e = self.embedder(timeEmbedding)
        

        x = self.conv1(input) 

        #print("--------------\nX :", x)
        #print("e :", e, "\n---------------------------")
        x = self.bNorm1(x)

        x = self.conv2(x)
        x = self.bNorm2(x)
        
        transformed = self.transformer(x)
        #print(transformed)
        return (x, transformed)

def buildUnet(latentSpaceSize) :
    noisedImages = kr.Input(shape=(32, 32, 1))
    noiseIntensity = kr.Input(shape=(1, 1, 1))

    embedding = kr.layers.Lambda(sinusoidalTimeEmbedding)(noiseIntensity)
    embedding = kr.layers.UpSampling2D(size=latentSpaceSize, interpolation="nearest")(embedding)

    initialConvLayer = kr.layers.Conv2D(64, kernel_size = 1)
    
    # Downsampling Layers
    convLeftLayer1 = UNetBlock(64, 32, upBlock = False)
    convLeftLayer2 = UNetBlock(128, 16, upBlock = False)
    convLeftLayer3 = UNetBlock(256, 8, upBlock = False)
    convLeftLayer4 = UNetBlock(512, 4, upBlock = False)

    # Upsampling Layers
    convRightLayer4 = UNetBlock(512, 4, upBlock = True)
    convRightLayer3 = UNetBlock(256, 8, upBlock = True)
    convRightLayer2 = UNetBlock(128, 16, upBlock = True)
    #convRightLayer1 = UNetBlock(64, 32, upBlock = True) ## SHOULD NOT BE CALLED

    # Final Layers
    conv1 = kr.layers.Conv2D(filters = 64, kernel_size = 1, strides = 1, padding="same", activation = kr.activations.relu)
    conv2 = kr.layers.Conv2D(filters = 64, kernel_size = 1, strides = 1, padding="same", activation = kr.activations.relu)
    finalLayer = kr.layers.Conv2D(filters = 1, kernel_size = 1, strides = 1, activation = kr.activations.relu)

    # Bottleneck
    bottleNeck = UNetBlock(1024, blockSize = 2, upBlock = True)  

    x = initialConvLayer(noisedImages)

    print("x is :")
    print(x)
    print(embedding)
    x = kr.layers.concatenate([x, embedding])

    layer1Skip, x = convLeftLayer1(x)
    layer2Skip, x = convLeftLayer2(x)
    layer3Skip, x = convLeftLayer3(x)
    layer4Skip, x = convLeftLayer4(x)

    _, x = bottleNeck(x) 

    x = kr.layers.concatenate([x,layer4Skip])
    _, x = convRightLayer4(x)

    x = kr.layers.concatenate([x,layer3Skip])
    _, x = convRightLayer3(x)
    x = kr.layers.concatenate([x,layer2Skip])
    _, x = convRightLayer2(x)

    x = kr.layers.concatenate([x,layer1Skip])

    x = conv1(x)
    x = conv2(x)
    x = finalLayer(x)

    return kr.Model([noisedImages, noiseIntensity], x)

class StableDiffusionModel(kr.Model) :
    """
    Implementation of a Stable Diffusion Model. 
    
    Uses the AutoEncoder class to generate and decode from the latent space.
    Uses the DiffusionModel class to generate new data from noised latent space.
    
    """
    

    
    def __init__(self, encoder, decoder, latentSize = 32, timeDim = 256, scheduleSteps = 1000, betaInitial = 0.02, betaFinal =1e-4):
        super().__init__()
        
        # Input Image
        self.encoder = encoder
        self.decoder = decoder
        self.myModel = buildUnet(32)
        
        self.latentSize = latentSize
        self.timeDim = timeDim

        self.scheduleSteps = scheduleSteps
        self.betaInitial = betaInitial
        self.betaFinal = betaFinal

        # Using a linear diffusion schedule:
        self.beta = tf.linspace(self.betaInitial, self.betaFinal, self.scheduleSteps)
        self.alpha = 1 - self.beta
        self.alphaHat = tf.math.cumprod(self.alpha)


    def noiseImage(self, image, step) : ########################################
      #print("StepShape:")
      #print(step.shape)
      sqrtAlphaHat = tf.math.sqrt(tf.gather(self.alphaHat, indices = step))[:,None, None]
      sqrtAlphaHatCompliment = tf.math.sqrt( tf.gather(1-self.alphaHat, indices=[step]))[:,None, None]
      noise = tf.random.normal(shape=())

      return image * sqrtAlphaHat + sqrtAlphaHatCompliment * noise, noise


    def getNoiseIntensity(self, diffusionTimes) :
        start = tf.acos(0.95)
        end = tf.acos(0.02)

        diffusion_angles = start + diffusionTimes * (end-start)
        signalIntensity = tf.cos(diffusion_angles)
        noiseIntensity = tf.sin(diffusion_angles)

        return noiseIntensity, signalIntensity
        

    def compile(self, **kwargs) :
      super().compile(**kwargs)

      self.lossMetric = tf.keras.metrics.Mean(name="loss")

    def calculatePrevStep(self, image, predictedNoise, noise,   step) :

      leftTerm = tf.math.pow(tf.math.sqrt(self.alpha[step]), -1)
      bracketTerm = (image - (tf.gather(1-self.alpha, indices=[step])/tf.gather(1-self.alphaHat, indices=[step])) * predictedNoise) #######################
      rightTerm = tf.math.sqrt(self.beta[step]) * noise

      return leftTerm*bracketTerm + rightTerm

    @property
    def metrics(self):
        return [self.lossMetric]

    def generateTimeSteps(self, size) :
        return tf.random.uniform(shape=(size,), dtype = tf.dtypes.int32, minval = 1, maxval = self.scheduleSteps)
    
    def denoise(self, noisedImages, noiseIntensity, signalIntensity) :
        predictedNoises = self.myModel.call([noisedImages, noiseIntensity**2])
        predictedImages = (noisedImages - (predictedNoises * noiseIntensity)) / signalIntensity
        
        return predictedNoises, predictedImages

        
    def train_step(self, images) :
        latentImages= self.encoder(images)
            
        
        noises = tf.random.normal(shape =(32, 32, 32, 1))
        diffusionTimes = tf.random.uniform(shape=(32, 1, 1, 1), minval=0.0, maxval=1.0)
        
        noiseIntensity, signalIntensity = self.getNoiseIntensity(diffusionTimes)
        
        noisedImages = noiseIntensity * noises + signalIntensity * latentImages
        
        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            predictedNoise, predictedImages = self.denoise(noisedImages, noiseIntensity, signalIntensity)

            noiseLoss = self.loss(noises, predictedNoise)  # used for training
        
        gradients = tape.gradient(noiseLoss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.lossMetric.update_state(noiseLoss)
        return {"loss" : self.lossMetric.result()}
    
    
    """
    