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




class UNetBlock(kr.layers.Layer) :
    def __init__(self, outputSize, blockSize = 32, upBlock = False, upSampler = ConvUpsample, downSampler = ConvDownsample) :
        super().__init__()
        self.blockSize = blockSize
        
        self.conv1 = kr.layers.Conv2D(outputSize, 3, padding = "same", activation = kr.activations.relu)
        self.bNorm1 = kr.layers.BatchNormalization()

        self.conv2 = kr.layers.Conv2D(outputSize, 3, padding = "same", activation = kr.activations.relu)
        self.bNorm2 = kr.layers.BatchNormalization()
        
        self.embedder = kr.layers.UpSampling2D(blockSize)
        self.embedderConv = kr.layers.Conv2D(outputSize, 3, padding = "same", activation = kr.activations.relu)
        
        # If upsampling occurs in block
        if (upBlock) :
            self.transformer = upSampler(outputSize)
        else :
            self.transformer = downSampler(outputSize)
    
    def call(self, inputs) :
        imageInput = inputs[0]
        timeEmbedding = inputs[1]
        
        #print("ImageIS")
        #print(imageInput)

        #print("PRE EMBEDDING:", timeEmbedding)
        e = self.embedder(timeEmbedding)
        #print("POST EMBEDDING:", e)
        #print("MY BLOCKSIZE: ", self.blockSize)
        e = self.embedderConv(e)

        x = self.conv1(imageInput) 

        #print("--------------\nX :", x)
        #print("e :", e, "\n---------------------------")
        x = self.bNorm1(x + e)

        x = self.conv2(x)
        x = self.bNorm2(x)
        
        transformed = self.transformer(x)
        #print(transformed)
        return (x, transformed)

        
class UNet(kr.layers.Layer):
    def __init__(self, blockDepths = [64, 128, 256, 512], blockSize = [32, 16, 8, 4], bottleNeck = 1024) :
        super().__init__()
        """
        self.layerCount = len(blockDepths)
        self.conv1 = kr.layers.Conv2D(filters = blockDepths[0], kernel_size = 1, strides = 1, padding="same", activation = kr.activations.relu)
        self.conv2 = kr.layers.Conv2D(filters = blockDepths[0], kernel_size = 1, strides = 1, padding="same", activation = kr.activations.relu)
        self.finalLayer = kr.layers.Conv2D(filters = 1, kernel_size = 1, strides = 1, activation = kr.activations.relu)
        self.bottleNeck = UNetBlock(bottleNeck, blockSize = blockSize[-1]//2, upBlock = True)
        
        
        self.downBlocks = []
        self.upBlocks = []

        self.downBlocks.append(UNetBlock(blockDepths[i], blockSize[i], upBlock = False))
        
        for i in range(self.layerCount) :
            self.downBlocks.append(UNetBlock(blockDepths[i], blockSize[i], upBlock = False))
            self.upBlocks.insert(0, UNetBlock(blockDepths[i], blockSize[i], upBlock = True))
        self.upBlocks.pop()


        _, lastBlock = self.bottleNeck((lastBlock, timeEmbedding))
        for i in range(self.layerCount-1) :
          lastBlock = kr.layers.concatenate((skipConnections[i], lastBlock))

          _, lastBlock = self.upBlocks[i]((lastBlock, timeEmbedding))
            
        
        lastBlock = self.conv1(lastBlock)
        lastBlock = self.conv2(lastBlock)
        return self.finalLayer(lastBlock)
        """

        # Downsampling Layers
        self.convLeftLayer1 = UNetBlock(64, 32, upBlock = False)
        self.convLeftLayer2 = UNetBlock(128, 16, upBlock = False)
        self.convLeftLayer3 = UNetBlock(256, 8, upBlock = False)
        self.convLeftLayer4 = UNetBlock(512, 4, upBlock = False)


        # Upsampling Layers
        self.convRightLayer4 = UNetBlock(512, 4, upBlock = True)
        self.convRightLayer3 = UNetBlock(256, 8, upBlock = True)
        self.convRightLayer2 = UNetBlock(128, 16, upBlock = True)
        self.convRightLayer1 = UNetBlock(64, 32, upBlock = True) ## SHOULD NOT BE CALLED

        # Final Layers
        self.conv1 = kr.layers.Conv2D(filters = blockDepths[0], kernel_size = 1, strides = 1, padding="same", activation = kr.activations.relu)
        self.conv2 = kr.layers.Conv2D(filters = blockDepths[0], kernel_size = 1, strides = 1, padding="same", activation = kr.activations.relu)
        self.finalLayer = kr.layers.Conv2D(filters = 1, kernel_size = 1, strides = 1, activation = kr.activations.relu)

        self.bottleNeck = UNetBlock(bottleNeck, blockSize = blockSize[-1]//2, upBlock = True)
            
    def call(self, input) :
        images, timeEmbedding = input

        print("IMAGE:")
        print(images)
        print("TIME EMBEDDING:")
        print(timeEmbedding)

        lastBlock = images
        skipConnections = []
        

        layer1Skip, x = self.convLeftLayer1((images, timeEmbedding))
        layer2Skip, x = self.convLeftLayer2((x, timeEmbedding))
        layer3Skip, x = self.convLeftLayer3((x, timeEmbedding))
        layer4Skip, x = self.convLeftLayer4((x, timeEmbedding))

        _, x = self.bottleNeck((x, timeEmbedding)) 

        x = kr.layers.concatenate([x,layer4Skip])
        _, x = self.convRightLayer4((x, timeEmbedding))

        x = kr.layers.concatenate([x,layer3Skip])
        _, x = self.convRightLayer3((x, timeEmbedding))

        x = kr.layers.concatenate([x,layer2Skip])
        _, x = self.convRightLayer2((x, timeEmbedding))

        x = kr.layers.concatenate([x,layer1Skip])
        #_, x = self.convRightLayer1((x, timeEmbedding))

        x = self.conv1(x)
        x = self.conv2(x)
        return self.finalLayer(x)


        
  

class StableDiffusionModel(kr.Model) :
    """
    Implementation of a Stable Diffusion Model. 
    
    Uses the AutoEncoder class to generate and decode from the latent space.
    Uses the DiffusionModel class to generate new data from noised latent space.
    
    """
    

    
############################### SIMPLY ADD TIME EMBEDDING AT EACH STEP!!!! ####
    def __init__(self, encoder, decoder, latentSize = 32, timeDim = 256, scheduleSteps = 1000, betaInitial = 0.02, betaFinal =1e-4):
        super().__init__()
        
        # Input Image
        self.encoder = encoder
        self.decoder = decoder
        self.UNet = UNet()
        
        self.latentSize = latentSize
        self.timeDim = timeDim
        self.scheduleSteps = scheduleSteps

        self.scheduleSteps = scheduleSteps
        self.betaInitial = betaInitial
        self.betaFinal = betaFinal

        # Using a linear diffusion schedule:
        self.beta = tf.linspace(self.betaInitial, self.betaFinal, self.scheduleSteps)
        self.alpha = 1 - self.beta
        self.alphaHat = tf.math.cumprod(self.alpha)


    def noiseImage(self, image, step) :
      #print("StepShape:")
      #print(step.shape)
      sqrtAlphaHat = tf.math.sqrt(tf.gather(self.alphaHat, indices = step))[:,None, None]
      sqrtAlphaHatCompliment = tf.math.sqrt( tf.gather(1-self.alphaHat, indices=[step]))[:,None, None]
      noise = tf.random.normal(shape=())

      return image * sqrtAlphaHat + sqrtAlphaHatCompliment * noise, noise
        

    def compile(self, **kwargs) :
      super().compile(**kwargs)

      self.lossMetric = tf.keras.metrics.Mean(name="loss")

    def calculatePrevStep(self, image, predictedNoise, noise,   step) :

      leftTerm = tf.math.pow(tf.math.sqrt(self.alpha[step]), -1)
      bracketTerm = (image - (tf.gather(1-self.alpha, indices=[step])/tf.gather(1-self.alphaHat, indices=[step])) * predictedNoise)
      rightTerm = tf.math.sqrt(self.beta[step]) * noise

      return leftTerm*bracketTerm + rightTerm

    @property
    def metrics(self):
        return [self.lossMetric]
        
    def sinusoidalTimeEmbedding(self, time) :
        channels = self.timeDim

        time = tf.cast(tf.expand_dims(time, axis=-1), dtype=tf.float32)

        frequencyInverse = tf.cast(1.0/(10000.0 ** (tf.range(0, channels, 2)/channels)), dtype=tf.float32)
        print("Time")
        print(time)
        posA = tf.sin(tf.tile(time, (1, channels // 2)) * frequencyInverse)
        posB = tf.cos(tf.tile(time, (1, channels // 2)) * frequencyInverse)
        a = tf.concat((posA, posB), axis=-1)[:, None, None]
        print(a.shape)
        return a

    def generateTimeSteps(self, size) :
        return tf.random.uniform(shape=(size,), dtype = tf.dtypes.int32, minval = 1, maxval = self.scheduleSteps)
    
    def denoise(self, noisedImage, noiseIntensity) :
        #print(noiseIntensity)
        noiseIntensity = tf.cast(noiseIntensity, dtype = tf.float32)
        timeEmbedding = self.sinusoidalTimeEmbedding(noiseIntensity)
        predictedNoise = self.UNet((noisedImage, timeEmbedding))
        predictedImage = (noisedImage - (predictedNoise * noiseIntensity)) / (1-noiseIntensity)

        return predictedNoise, predictedImage

        
    def train_step(self, images) :
        latentImages= self.encoder(images)

        #print("LATENT IMAGES")
        #print(latentImages)
        
        steps = self.generateTimeSteps(latentImages.shape[0])

        #print("STEPS")
        #print(steps)

        noisedImages, noises = self.noiseImage(latentImages, steps)

        #print("NOISED IMAGE:")
        #print(noisedImages)
        
        with tf.GradientTape() as tape:
            # Network predictes noise and image.
            predictedNoises, predictedImages = self.denoise(noisedImages, steps)
            noiseLoss = self.loss(noises, predictedNoises)
        
        gradients = tape.gradient(noiseLoss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

        self.lossMetric.update_state(noiseLoss)
        return {"loss" : self.lossMetric.result()}
    