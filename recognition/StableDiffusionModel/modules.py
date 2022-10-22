# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:47:08 2022

@author: Daniel Ju Lian Wong
"""


import numpy as np

import tensorflow.keras.layers as nn
import tensorflow_addons as tfa
import tensorflow.keras as kr
import tensorflow as tf

import math
from inspect import isfunction

from tensorflow import einsum
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer
from einops import rearrange
from einops.layers.tensorflow import Rearrange
from functools import partial

################################ CONSTANTS ##################################

# Precalculated values:
TIME_STEPS = 200
BETA = np.linspace(0.0001, 0.02, TIME_STEPS)
ALPHA = 1 - BETA
ALPHA_HAT = np.cumprod(ALPHA, 0)
ALPHA_HAT = np.concatenate((np.array([1.]), ALPHA_HAT[:-1]), axis=0)
SQRT_ALPHA_HAT = np.sqrt(ALPHA_HAT)
SQRT_ALPHA_HAT_COMPLIMENT = np.sqrt(1-ALPHA_HAT)

TARGET_SIZE = (32, 32)
CHANNELS = 1
BATCH_SIZE=64
TIME_STEPS = 200
EPOCHS = 10

############################### HELPER LAYERS #################################


class ZeroPaddedConv2D(kr.layers.Layer) :
    """
    2D Convolution layer with custom zero-padding
    """
    
    def __init__(self, filters, kernelSize = 3, stride = (1,1), padding = (1,1), activation = None) :
        """
        Create a new padded convolutional layer
        
        Parameters:
            filters (int): number of filters used.
            stride (int, list(int)): stride used in convolution
            padding (int, list(int)): zero padding added
        """
        super().__init__()
        self.padd = kr.layers.ZeroPadding2D(padding = padding)
        self.conv = kr.layers.Conv2D(filters, kernelSize, stride, activation = activation)
        
    def call(self, inputs) :
        return self.conv(self.padd(inputs))
        

class ConvDownsample(kr.layers.Layer) :
    """
    Conv2D layer with preset parameters to downsize image by 2. Also performs
    a batch normalization.
    """
    def __init__(self, outputLayers, activation = None):
        """
        Constructs a new downsampling convolutional layer
        Parameters:
            activation (kr.activations): activation function of convolutional layer
        """
        super().__init__()
        self.conv1 = ZeroPaddedConv2D(outputLayers, 
                                         kernelSize = 3, 
                                         stride = (2,2), 
                                         padding = (1,1), 
                                         activation = activation)
        self.norm1 = kr.layers.BatchNormalization()
        
    def call(self, inputs) :
        return self.norm1(self.conv1(inputs))
    

class ConvUpsample(kr.layers.Layer) :
    """
    Conv2D layer with preset parameters to upsize image by 2. Also performs
    a batch normalization.
    """
    def __init__(self, outputSize):
        """
        Constructs a new upsampling convolutional layer
        Parameters:
            activation (kr.activations): activation function of convolutional layer
        """
        super().__init__()
        self.transform = kr.layers.Conv2DTranspose(outputSize, 3, 2, padding="same")
        self.norm1 = kr.layers.BatchNormalization()
        
    def call(self, inputs) :
        return self.norm1(self.transform(inputs))
        
        

class ResidualNetBlock(kr.layers.Layer) :
    """ 
    Convolutional block with a skip connection attached 
    """
    def __init__(self, 
                 inputDim, 
                 outputDim, 
                 kernelSize, 
                 normLayers = True, 
                 activation = None, 
                 epsilon = 1e-4) :
        """
        Constructs a new ResidualNetBlock
        Parameters:
            inputDim (int): Number of input channels
            outputDim (int): Number of output channels
            kernelSize (int): size of the kernel in the convole layers
            activation (kr.activations): activation function of conv layers
            epsilon (float): epislon of internal layers
        """
        super().__init__()
        # If true, then batchNorm layers are applied
        self.__isNormed = normLayers
        
        self.conv1 = ZeroPaddedConv2D(outputDim, kernelSize, activation = activation)
        self.norm1 = kr.layers.BatchNormalization()
        
        if (inputDim != outputDim) :
            # If the input dimension is different from the output dimension 
            # Convolution with kenrel size 1 and outputDim many filters
            # is used to resize data
            self.skip1 = kr.layers.Conv2D(outputDim, 1)
        else :
            # Else, input is summed, like standard skip connection.
            self.skip1 = kr.layers.Lambda(lambda x : x)
        
        
    def call(self, inputs) :
        if (self.__isNormed) :
            x = self.conv1(inputs)
            x = x + self.skip1(inputs)
            
            return self.norm1(x)
        else : # If groupNorm layers are not enabled
            x = self.conv1(inputs)
            x = self.conv2(x)
            
            return x + self.skip1(inputs)

########################## AUTOENCODER LAYERS #################################
        
class Encoder(kr.Sequential) :
    """
    Convolutional encoder used to turn image into its latent representation in the autoencoder. 
    Includes residual and convolutional downsampling layers.
    """
    def __init__(self,  activation = kr.activations.swish, downSampleLayer = ConvDownsample, normLayers = True) :
        """
        Constructs a new Encoder
        Parameters:
            activation (kr.activations): activation function used by internal layers
            downSampleLayer (kr.layers.Layer): Layer used to sample down
            normLayers (bool): True if batchNorm layers in residual blocks are enabled, 
                                false otherwise
        """
        super().__init__(
            [
            # Block downsampling by factor of 2
                ResidualNetBlock(1, 256, 3, activation = activation, normLayers = normLayers),
                ResidualNetBlock(256, 256, 3, activation = activation, normLayers = normLayers), 
                downSampleLayer(256),
            
            # Block downsampling by factor of 2
                ResidualNetBlock(256, 256, 3, activation = activation, normLayers = normLayers),  
                ResidualNetBlock(256, 256, 3, activation = activation, normLayers = normLayers), 
                downSampleLayer(256),
            
            # Reducing filter channel back to 1
                ResidualNetBlock(256, 256, 3, activation = activation, normLayers = normLayers),   
                ResidualNetBlock(256, 1, 3, activation = activation, normLayers = normLayers)
            ])
        
class Decoder(kr.Sequential) :
    """
    Convolutional decoder used to approximate the original image from its latent representation 
    Includes residual and convolutional downsampling layers.
    """
    def __init__(self, activation = kr.activations.swish, upSampleLayer = ConvUpsample, normLayers = True) :
        """
        Constructs a new Decoder
        Parameters:
            activation (kr.activations): activation function used by internal layers
            upSampleLayer (kr.layers.Layer): Layer used to sample up
            normLayers (bool): True if batchNorm layers in residual blocks are enabled, 
                                false otherwise
        """
        super().__init__(
            [    
            # Block upsampling by factor of 2
                ResidualNetBlock(1, 256, 3, activation = activation, normLayers = normLayers),
                ResidualNetBlock(256, 256, 3, activation = activation, normLayers = normLayers),   
                upSampleLayer(256),
            
            # Block upsampling by factor of 2
                ResidualNetBlock(256, 256, 3, activation = activation, normLayers = normLayers),   
                ResidualNetBlock(256, 256, 3, activation = activation, normLayers = normLayers),   
                upSampleLayer(256),

            # Reducing filter channel back to 1
                ResidualNetBlock(256, 128, 3, activation = activation, normLayers = normLayers),  
                ResidualNetBlock(128, 64, 3, activation = activation, normLayers = normLayers),
                ResidualNetBlock(64, 1, 3, activation = activation, normLayers = normLayers)
            ])


                                 
class AutoEncoder(kr.Model) :
    """
    Comprised of an encoder and a decoder. Converts an image to its latent representation,
    then approximates the image based off of that latent representation.
    """
    
    def __init__(self, inputSize, latentSpaceSize, activation = kr.activations.swish, normLayers = True) :
        """
        Constructs a new AutoEncoder
        
        Parameters:
            inputSize (int): size of the input image, e.g. 64 for 64x64
            latentSpaceSize (int): size of the latent space image, e.g. 32 for 32x32
            activation (kr.activations.Activation): activation layer applied on 
                                                    convolutions in autoencoder
            normLayers (bool): True if batchNorm layers in residual blocks are enabled, 
                                false otherwise
        """
        super().__init__()
        
        self.inputSize = inputSize
        self.latentSpaceSize = latentSpaceSize
        
        
        # Builds an encoder submodel
        self.encoder = self.__buildEncoderLayers(activation=activation, 
                                                 normLayers=normLayers)
        
        
        # Builds a decoder submodel
        self.decoder = self.__buildDecoderLayers(activation=activation, 
                                                 normLayers = normLayers)
        
    def call(self, inputs) :
        x = self.encoder(inputs)
        return self.decoder(x)
        
    def __buildEncoderLayers(self,
                             activation=kr.activations.swish, 
                             normLayers = True) : 
        """
        Builds a new Encoder instance
        
        Parameters:
            inputSize (int): size of the input image, e.g. 64 for 64x64
            activation (kr.activations.Activation): activation layer applied on 
                                                    convolutions in autoencoder
            normLayers (bool): True if batchNorm layers in residual blocks are enabled, 
                                false otherwise
        """
        
        return Encoder(activation=activation, normLayers = normLayers)
        
        
    def __buildDecoderLayers(self, 
                             activation=kr.activations.swish, 
                             normLayers = True) :
        """
        Builds a new Decoder instance
        
        Parameters:
            activation (kr.activations.Activation): activation layer applied on 
                                                    convolutions in autoencoder
            normLayers (bool): True if batchNorm layers in residual blocks are enabled, 
                                false otherwise
        """
        return Decoder(activation=activation, normLayers = normLayers)
        

    def buildEncoder(self, dims) :
        """
        Returns a new model from the encoder inside the autoencoder
        """
        newInput = kr.Input((dims, dims, 1))
        return kr.models.Model(newInput, self.encoder(newInput)) 
    
    def buildDecoder(self, dims) :
        """
        Returns a new model from the decoder inside the autoencoder
        """
        newLatent = kr.Input((dims, dims, 1))
        return kr.models.Model(newLatent, self.decoder(newLatent)) 
    
    
################################################################################

""" 
Code past this point is derivative of tutorial from:
https://medium.com/@vedantjumle/image-generation-with-diffusion-models-using-keras-and-tensorflow-9f60aae72ac

"""

def preprocess(x, y):
    """ casts and normalises"""
    return tf.image.resize(tf.cast(x, tf.float32) / 127.5 - 1, (32, 32))

def setSeed(key):
    """ resets numpy random number generation """
    np.random.seed(key)


def addNoise(key, input, t):
    """ Adds noise corresponding to the current timestamp"""
    setSeed(key)
    noise = np.random.normal(size=input.shape)
    imageIntensity = np.reshape(np.take(SQRT_ALPHA_HAT, t), (-1, 1, 1, 1))
    noiseIntensity = np.reshape(np.take(SQRT_ALPHA_HAT_COMPLIMENT, t), (-1, 1, 1, 1))
    noisedImage = imageIntensity  * input + noiseIntensity  * noise
    return noisedImage, noise


def createTimeStamp(key, num):
    """Intialises random ints to sample random timestamps"""
    setSeed(key)
    return tf.random.uniform(shape=[num], minval=0, maxval=TIME_STEPS, dtype=tf.int32)


def exists(input):
    """Checks if the input exists, returning true if it does and else otherwise"""
    return input is not None

def default(val, d):
    """ Calls d if it is a function """
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(kr.layers.Layer):
    """
    Wrapper class that makes input layer residual
    """
    def __init__(self, fn):
        """
        Wrapper class that makes input layer residual
        """
        super(Residual, self).__init__()
        self.fn = fn

    def call(self, input, training=True):
        return self.fn(input, training=training) + input

class Identity(kr.layers.Layer):
    """
    A simple identity layer
    """
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x, training=True):
        return tf.identity(x)

def Upsample(dim):
    """
    Builds and returns a convolutional upsampling layer
    """
    return nn.Conv2DTranspose(filters=dim, kernel_size=4, strides=2, padding='SAME')

def Downsample(dim):
    """
    Builds and returns a convolutional downsampling layer
    """
    return nn.Conv2D(filters=dim, kernel_size=4, strides=2, padding='SAME')

class SinusoidalPosEmb(kr.layers.Layer):
    """ 
    Helper class for constructing timestamp using suinsodial 
    position embedding
    """
    def __init__(self, dim, maxPos=10000):
        """
        Constructs a new posEmbedding helper
        Parameters
          maxPos (int): the max Position
          dim (int): the number of dimensions
        """
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.maxPos = maxPos

    def call(self, x, training=True):
        x = tf.cast(x, tf.float32)
        hDim = self.dim // 2
        e = math.log(self.maxPos) / (hDim - 1)
        e = tf.exp(tf.range(hDim, dtype=tf.float32) * -e)
        e = x[:, None] * e[None, :]

        e = tf.concat([tf.sin(e), tf.cos(e)], axis=-1)

        return e

class PreNorm(kr.layers.Layer):
    """
    Layer that applies group normalisation to its input 
    """
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.normal = tfa.layers.GroupNormalization(dim)

    def call(self, x, training=True):
        x = self.normal(x)
        return self.fn(x)

class SiLU(kr.layers.Layer):
    """
    Layer that performs SiLU activation
    """
    def __init__(self):
        super(SiLU, self).__init__()

    def call(self, x, training=True):
        return x * tf.nn.sigmoid(x)

def gelu(x, approximate=False):
    """
    Calculates the Gaussian Error Linear Units (GeLU)
    """
    if approximate:
        coeff = tf.cast(0.044715, x.dtype)
        return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
    else:
        return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

class GELU(kr.layers.Layer):
    """
    Layer that performs gelu activation
    """

    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        return gelu(x, self.approximate)


class Block(kr.layers.Layer):
    """
    Core building block of the unet Model. 
    Applies groupNormalisation and convolutions
    """
    def __init__(self, dim, groups=8):
        super(Block, self).__init__()
        self.projection = nn.Conv2D(dim, kernel_size=3, strides=1, padding='SAME')
        self.normal = tfa.layers.GroupNormalization(groups, epsilon=1e-05)
        self.activation = SiLU()


    def call(self, input, gammaBeta=None, training=True):
        x = self.projection(input)
        x = self.normal(x, training=training)

        if exists(gammaBeta):
            gamma, BETA = gammaBeta
            x = x * (gamma + 1) + BETA

        return self.activation(x)

class ResnetBlock(kr.layers.Layer):
    def __init__(self, dim, outputDim, timeEmbedDim=None, groups=8):
        """
        Layer which wraps multiple Block layers, injects time embedding, 
        and has skip connections
        """
        super(ResnetBlock, self).__init__()

        self.mlp = Sequential([
            SiLU(),
            nn.Dense(units=outputDim * 2)
        ]) if exists(timeEmbedDim) else None

        self.block1 = Block(outputDim, groups=groups)
        self.block2 = Block(outputDim, groups=groups)
        self.resConv = nn.Conv2D(filters=outputDim, kernel_size=1, strides=1) if dim != outputDim else Identity()

    def call(self, x, embeddingTime=None, training=True):
        gammaBeta = None
        if exists(self.mlp) and exists(embeddingTime):
            embeddingTime = self.mlp(embeddingTime)
            embeddingTime = rearrange(embeddingTime, 'b c -> b 1 1 c')
            gammaBeta = tf.split(embeddingTime, num_or_size_splits=2, axis=-1)

        h = self.block1(x, gammaBeta=gammaBeta, training=training)
        h = self.block2(h, training=training)

        return h + self.resConv(x)

class LinearAttention(kr.layers.Layer):
    """
    Linear Attention block that injects global context
    """
    def __init__(self, dim, heads=4, dimHead=32):
        super(LinearAttention, self).__init__()
        self.scale = dimHead ** -0.5
        self.heads = heads
        self.hidden_dim = dimHead * heads

        self.attend = nn.Softmax()
        self.queryKeyValue = nn.Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)

        self.outputLayer = Sequential([
            nn.Conv2D(filters=dim, kernel_size=1, strides=1),
            tfa.layers.GroupNormalization(dim, epsilon=1e-05)
        ])

    def call(self, input, training=True):
        b, h, w, c = input.shape
        queryKeyValue = self.queryKeyValue(input)
        queryKeyValue = tf.split(queryKeyValue, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), queryKeyValue)

        q = tf.nn.softmax(q, axis=-2)
        k = tf.nn.softmax(k, axis=-1)

        q = q * self.scale
        context = einsum('b h d n, b h e n -> b h d e', k, v)

        out = einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b x y (h c)', h=self.heads, x=h, y=w)
        out = self.outputLayer(out, training=training)

        return out

class Attention(kr.layers.Layer):
    """
    Attention block that injects global context
    """
    def __init__(self, dim, heads=4, dimHead=32):
        super(Attention, self).__init__()
        self.scale = dimHead ** -0.5
        self.heads = heads
        self.hidden_dim = dimHead * heads

        self.queryKeyValue = nn.Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)
        self.outputLayer = nn.Conv2D(filters=dim, kernel_size=1, strides=1)

    def call(self, x, training=True):
        b, h, w, c = x.shape
        queryKeyValue = self.queryKeyValue(x)
        queryKeyValue = tf.split(queryKeyValue, num_or_size_splits=3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), queryKeyValue)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        sim_max = tf.stop_gradient(tf.expand_dims(tf.argmax(sim, axis=-1), axis=-1))
        sim_max = tf.cast(sim_max, tf.float32)
        sim = sim - sim_max
        attn = tf.nn.softmax(sim, axis=-1)

        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x = h, y = w)
        out = self.outputLayer(out, training=training)

        return out

class Unet(Model):
    """
    Unet model which is run 
    inside of the autoEncoder 
    for the stable diffusion model
    """
    def __init__(self,
                 dim=64,
                 initialDim=None,
                 outputDim=None,
                 dimMultipliers=(1, 2, 4, 8),
                 channels=1,
                 normGroups=8,
                 learningVar=False,
                 isEmbeddingTime=True
                 ):
        super(Unet, self).__init__()
        
        # determine dimensions
        self.channels = channels
        
        initialDim = default(initialDim, dim // 3 * 2)
        self.initialConv = nn.Conv2D(filters=initialDim, kernel_size=7, strides=1, padding='SAME')
        
        dims = [initialDim, *map(lambda m: dim * m, dimMultipliers)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        wrappedBlock = partial(ResnetBlock, groups = normGroups)
        
        # time eeddings
        time_dim = dim * 4
        self.isEmbeddingTime = isEmbeddingTime
        
        self.timeEmbDense = Sequential([
            SinusoidalPosEmb(dim),
            nn.Dense(units=time_dim),
            GELU(),
            nn.Dense(units=time_dim)
        ], name="time eeddings")
        
        # layers
        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)
        
        for ind, (dim_in, outputDim) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append([
                wrappedBlock(dim_in, outputDim, timeEmbedDim=time_dim),
                wrappedBlock(outputDim, outputDim, timeEmbedDim=time_dim),
                Residual(PreNorm(outputDim, LinearAttention(outputDim))),
                Downsample(outputDim) if not is_last else Identity()
            ])
  
        middleDim = dims[-1]
        self.midBlock1 = wrappedBlock(middleDim, middleDim, timeEmbedDim=time_dim)
        self.midAttention = Residual(PreNorm(middleDim, Attention(middleDim)))
        self.midBlock2 = wrappedBlock(middleDim, middleDim, timeEmbedDim=time_dim)
        
        for ind, (dim_in, outputDim) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append([
                wrappedBlock(outputDim * 2, dim_in, timeEmbedDim=time_dim),
                wrappedBlock(dim_in, dim_in, timeEmbedDim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else Identity()
            ])
        
        
        self.finalConv = Sequential([
            wrappedBlock(dim * 2, dim),
            nn.Conv2D(filters=1, kernel_size=1, strides=1)
        ], name="output")
        
    def call(self, x, time=None, training=True, **kwargs):
        x = self.initialConv(x)
        t = self.timeEmbDense(time)
        
        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.midBlock1(x, t)
        x = self.midAttention(x)
        x = self.midBlock2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = tf.concat([x, h.pop()], axis=-1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = tf.concat([x, h.pop()], axis=-1)
        x = self.finalConv(x)
        return x