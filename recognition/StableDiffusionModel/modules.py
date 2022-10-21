# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:47:08 2022

@author: Daniel Ju Lian Wong
"""


#################### HELPER LAYERS #######################


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
    def __init__(self, inputDim, outputDim, kernelSize, normLayers = True, activation = None, epsilon = 1e-4) :
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
        self.encoder = self.__buildEncoderLayers(inputSize, 
                                                 activation=activation, 
                                                 normLayers=normLayers)
        
        
        # Builds a decoder submodel
        self.decoder = self.__buildDecoderLayers(latentSpaceSize, inputSize, activation=activation, normLayers = normLayers)
        
    def call(self, inputs) :
        x = self.encoder(inputs)
        return self.decoder(x)
        
    def __buildEncoderLayers(self, 
                             inputSize, 
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
        
        return Encoder(inputSize, activation=activation, normLayers = normLayers)
        
        
    def __buildDecoderLayers(self, 
                             latentSpaceSize, 
                             outputSize, 
                             activation=kr.activations.swish, 
                             normLayers = True) :
        """
        Builds a new Decoder instance
        
        Parameters:
            latentSpaceSize (int): size of the latent space image, e.g. 32 for 32x32
            activation (kr.activations.Activation): activation layer applied on 
                                                    convolutions in autoencoder
            normLayers (bool): True if batchNorm layers in residual blocks are enabled, 
                                false otherwise
        """
        return Decoder(latentSpaceSize, outputSize, activation=activation, normLayers = normLayers)
        

    def buildEncoder(self) :
        """
        Returns a new model from the encoder inside the autoencoder
        """
        newInput = kr.Input(self.inputSize)
        return kr.models.Model(newInput, self.encoder((newInput, newInput, 1))) 
    
    def buildDecoder(self) :
        """
        Returns a new model from the decoder inside the autoencoder
        """
        newLatent = kr.Input(self.latentSpaceSize)
        return kr.models.Model(newLatent, self.decoder((newInput, newInput, 1))) 
#####################