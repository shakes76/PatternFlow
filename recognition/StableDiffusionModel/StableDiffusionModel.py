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



class SinusodialTimeEmbedding(kr.layers.Layer) :
  def __init__(self, dimension, maxPos = 10000):
    super().__init__()

    self.dimension = dimension
    self.maxPos = maxPos

  def call(self, input, training=True) :
    # Casting to float to allow later operations (i.e. mult)
    hDim = ((self.dim // 2) - 1)

    x = tf.cast(input, tf.float32)
    e = tf.math.log(self.maxPos) / hDim
    e = tf.exp(tf.range(hDim, dtype=tf.float32) * -e)
    e = x[:, None] * e[None, :]

    e = tf.concat([tf.sin(e), tf.cos(e)], axis=-1)

#
class ResidualLayer(kr.layers.Layer) :
  def __init__(self, function):
    self.function = function


  def call(self, x, training=True):
    return self.function(x, training=training) + x

class PreNorm(kr.layers.Layer):
  def __init__(self, dim, fn):
    super(PreNorm, self).__init__()
    self.fn = fn
    self.norm = kr.layers.LayerNormalization(dim)

  def call(self, x, training=True):
    x = self.norm(x)
    return self.fn(x)

class GELU(kr.layers.Layer):
    def __init__(self):
        super(GELU, self).__init__()

    def call(self, input, training=True):
      return 0.5 * input * (1.0 + tf.math.erf(input / tf.cast(1.4142135623730951, input.dtype)))


class Block(kr.layers.Layer):
    def __init__(self, dim, groups=8):
        super().__init__()
        self.projection = kr.layers.Conv2D(dim, kernel_size=3, strides=1, padding='SAME')
        self.norm = tfa.layers.GroupNormalization(groups, epsilon=1e-05)
        self.act = tf.nn.silu()


    def call(self, input, gammaBeta=None, training=True):
        x = self.proj(input)
        x = self.norm(x, training=training)

        if gammaBeta is not None:
            gamma, beta = gammaBeta
            x = x * (gamma + 1) + beta

        x = self.act(x)
        return x
    
class ResidualBlock(kr.layers.Layer):
    def __init__(self, channelsIn, channelsOut, timeEmbeddingDim=None, groups=8):
        super().__init__()

        # Multilayer perceptron injecting embedding
        self.mlp = kr.Sequential([
            kr.nn.silu(),
            kr.nn.Dense(units=timeEmbeddingDim * 2)
        ]) if timeEmbeddingDim is not None else None

        self.block1 = Block(channelsOut, groups=groups)
        self.block2 = Block(channelsOut, groups=groups)
        self.conv1 = kr.layers.Conv2D(filters=channelsOut, kernel_size=1, strides=1) if channelsIn != channelsOut else IdentityLayer()

    def call(self, input, timeEmbedding=None, training = True):
      gammaBeta = None

      if (self.mlp is not None and timeEmbedding is not None) :
        timeEmbedding = self.mlp(timeEmbedding)
        timeEmbedding = rearrange(timeEmbedding, 'b c -> b 1 1 c')
        gammaBeta = tf.split(timeEmbedding, num_or_size_splits=2, axis=-1)

      x = self.block1(input, gamma_beta=gammaBeta, training=training)
      x = self.block2(x, training=training)

      return x + self.conv1(input)

class LinearAttention(kr.layers.Layer):
    def __init__(self, dim, heads=4, headDim=32):
        super(LinearAttention, self).__init__()
        self.scale = headDim ** -0.5
        self.heads = heads
        self.hiddenDim = headDim * heads

        self.attend = kr.nn.Softmax()
        self.ConvToQueryKeyVal = kr.layers.Conv2D(filters=self.hiddenDim * 3, kernel_size=1, strides=1, use_bias=False)

        self.output = kr.Sequential([
            kr.layers.Conv2D(filters=dim, kernel_size=1, strides=1),
            kr.layers.LayerNormalization(dim)
        ])


    ## TODO: rewrite later, this is maybe a bit verbose.
    def call(self, x, training=True):
        b, h, w, c = x.shape
        queryKeyVal = self.ConvToQueryKeyVal(x)
        queryKeyVal = tf.split(queryKeyVal, num_or_size_splits=3, axis=-1)
        query, key, val = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), queryKeyVal)

        query = tf.nn.softmax(query, axis=-2)
        key = tf.nn.softmax(key, axis=-1)

        query = query * self.scale
        context = tf.einsum('b h d n, b h e n -> b h d e', key, val)

        out = tf.einsum('b h d e, b h d n -> b h e n', context, query)
        out = rearrange(out, 'b h c (x y) -> b x y (h c)', h=self.heads, x=h, y=w)
        out = self.output(out, training=training)
        return out





class UNetBlock(kr.layers.Layer) :
    def __init__(self, outputSize, blockSize = 32, upBlock = False, upSampler = ConvUpsample, downSampler = ConvDownsample) :
        super().__init__()
        self.blockSize = blockSize
        self.outputSize = outputSize
        
        self.conv1 = kr.layers.Conv2D(outputSize, 3, padding = "same", activation = kr.activations.swish)
        self.bNorm1 = tfa.layers.GroupNormalization(groups = 8)

        self.conv2 = kr.layers.Conv2D(outputSize, 3, padding = "same", activation = kr.activations.swish)
        self.bNorm2 = tfa.layers.GroupNormalization(groups = 8)

        # If upsampling occurs in block
        if (upBlock) :
            self.transformer = upSampler(outputSize)
        else :
            self.transformer = downSampler(outputSize)
    
    def call(self, input) :
        

        x = self.conv1(input) 
        x = self.bNorm1(x)

        x = self.conv2(x)
        x = self.bNorm2(x)
        
        if (input.shape[-1] == x.shape[-1]) :
          x = (x + input)
        else :
          x = kr.layers.concatenate([x, input])

        transformed = self.transformer(x)
        #print(transformed)

        return (x,transformed)
    
class Attention(kr.layers.Layer):
    def __init__(self, dim, heads=4, headDim=32):
        super(Attention, self).__init__()
        self.scale = headDim ** -0.5
        self.heads = heads
        self.hiddenDim = headDim * heads

        self.convQueryKeyVal = kr.layers.Conv2D(filters=self.hiddenDim * 3, kernel_size=1, strides=1, use_bias=False)
        self.conv = kr.layers.Conv2D(filters=dim, kernel_size=1, strides=1)

    def call(self, x, training=True):
        b, h, w, c = x.shape
        queryKeyValue = self.convQueryKeyVal(x)
        queryKeyValue = tf.split(queryKeyValue, num_or_size_splits=3, axis=-1)
        query, key, value = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), queryKeyValue)
        query = query * self.scale

        sim = tf.einsum('b h d i, b h d j -> b h i j', query, key)
        sMax = tf.stop_gradient(tf.expand_dims(tf.argmax(sim, axis=-1), axis=-1))
        sMax = tf.cast(sMax, tf.float32)
        sim = sim - sMax
        attn = tf.nn.softmax(sim, axis=-1)

        o = tf.einsum('b h i j, b h d j -> b h i d', attn, value)
        o = rearrange(o, 'b h (x y) d -> b x y (h d)', x = h, y = w)
        o = self.conv(o, training=training)

        return o


def buildUnet(latentSpaceSize) :
    noisedImages = kr.Input(shape=(32, 32, 1))
    noiseIntensity = kr.Input(shape=(1, 1, 1))

    embedding = kr.layers.Lambda(sinusoidalTimeEmbedding)(noiseIntensity)
    embedding1 = kr.layers.UpSampling2D(size=latentSpaceSize, interpolation="nearest")(embedding)
    embedding2 = kr.layers.UpSampling2D(size=(32, 32), interpolation="nearest")(embedding)
    embedding2 = kr.layers.UpSampling2D(size=(32, 32), interpolation="nearest")(embedding)

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
    conv1 = kr.layers.Conv2D(filters = 64, kernel_size = 1, strides = 1, padding="same", activation = kr.activations.swish)
    conv2 = kr.layers.Conv2D(filters = 64, kernel_size = 1, strides = 1, padding="same", activation = kr.activations.swish)
    conv3 = kr.layers.Conv2D(filters = 32, kernel_size = 1, strides = 1, padding="same", activation = kr.activations.swish)
    finalLayer = kr.layers.Conv2D(filters = 1, kernel_size = 1, strides = 1, activation = kr.activations.swish)

    bNorm1 = tfa.layers.GroupNormalization(groups = 8)
    bNorm2 = tfa.layers.GroupNormalization(groups = 8)
    bNorm3 = tfa.layers.GroupNormalization(groups = 8)

    # Bottleneck
    bottleNeck = UNetBlock(1024, blockSize = 2, upBlock = True)  

    x = initialConvLayer(noisedImages)
    x = kr.layers.concatenate([x, embedding1])

    # Concatenating embedding onto topmost layer
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

    x = kr.layers.concatenate([x, layer1Skip])

    x = conv1(x)
    x = bNorm1(x)
    x = conv2(x)
    #x = bNorm2(x)
    x = conv3(x)
    #x = bNorm3(x)
    x = finalLayer(x)

    return kr.Model([noisedImages, noiseIntensity], x)

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



class DiffusionModel(kr.Model) :
    def __init__(self, betaMin = 0.0001, betaMax = 0.02, timeSteps = 200):
        super().__init__()
        self.betaSchedule = tf.linspace(betaMin, betaMax, timeSteps)
        self.timeSteps = timeSteps
        self.alpha = 1- self.betaSchedule
        self.alphaHat = tf.math.cumprod(self.alpha, 0)
        self.alphaHat = tf.concat((tf.convert_to_tensor([1.]), self.alphaHat[:-1]), axis=0)
        self.sqrtAlphaHat = tf.math.sqrt(self.alphaHat)
        self.sqrtAlphaHatCompliment = tf.math.sqrt(1-self.alphaHat)


    def addNoise(self, input, step) :
        noise = tf.random.normal(shape=input.shape)
        reshapedSAH = tf.reshape(self.sqrtAlphaHat[step], (-1, 1, 1, 1))
        reshapedSAHC = tf.reshape(self.sqrtAlphaHatCompliment[step], (-1, 1, 1, 1))
        noisedInput = reshapedSAH  * input + reshapedSAHC  * noise
        
        return noisedInput, noise

    def generateTimeSteps(self, stepsGenerated):
        return tf.random.uniform(shape=[stepsGenerated], minval=0, maxval=self.timeSteps, dtype=tf.int32)

testModel = DiffusionModel()
print(testModel.generateTimeSteps(100).shape)


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
    