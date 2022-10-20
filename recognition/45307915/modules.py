import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Dropout, Add, UpSampling2D, Concatenate, Activation
from tensorflow.keras.models import Model

import tensorflow_addons as tf
from tensorflow_addons.layers import InstanceNormalization

class ImprovedUNETModel():
    
    def __init__(self, img_width, img_height):
        """ Create a new Improved UNET Model instance.
        
        Described in the paper Brain Tumor Segmentation and Radiomics
        
        Parameters:
            img_wdith (int): Image Width
            img_height (int): Image Height
        """
        self.img_width = img_width
        self.img_height = img_height
        
        self.init_filters = 16
        self.padding = 'same'
        self.leakyAlpha = 1e-2
        self.dropoutRate = 0.3
        
    def contextModule(self, inputs, n_filters):
        """ Improved UNET Context Block
        
        "Each context module is in fact a pre-activation residual block with two
        3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between"
        
        Batch normalisation layers were added
        Leaky ReLU activations were used

        Parameters:
            inputs (tf.Tensor): A () tensor inputted into the module
            n_filters (int): Number of filters for this module

        Return:
            tf.Tensor: A () tensor output from this module

        """
        x = InstanceNormalization()(inputs)
        x = LeakyReLU(alpha=self.leakyAlpha)(x)
        x = Conv2D(n_filters, (3,3), padding=self.padding)(x)
        
        x = Dropout(self.dropoutRate)(x)
        
        x = InstanceNormalization()(x)
        x = LeakyReLU(alpha=self.leakyAlpha)(x)
        x = Conv2D(n_filters, (3,3), padding=self.padding)(x)
        
        return x
    
    def upsamplingModule(self, inputs, n_filters):
        """ Improved UNET Upsampling Block
        
        "This is achieved by first upsampling the low resolution feature maps, 
        which is done by means of a simple upscale that repeats the feature voxels 
        twice in each spatial dimension, followed by a 3x3x3 convolution"

        Parameters:
            inputs (tf.Tensor): A () tensor inputted into the module 
            n_filters (int): Number of filters for this module

        Return:
            tf.Tensor: A () tensor output from this module

        """
        x = UpSampling2D(interpolation='bilinear')(inputs)
        x = Conv2D(n_filters, (3,3), padding=self.padding)(x)
        x = LeakyReLU(alpha=self.leakyAlpha)(x)
        x = InstanceNormalization()(x)
        
        return x
    
    def localisationModule(self, inputs, n_filters):
        """ Improved UNET Localisation Layer
        
        "A localization module consists of a 3x3x3 convolution followed 
        by a 1x1x1 convolution"

        Parameters:
            inputs (tf.Tensor): A () tensor inputted into the module 
            n_filters (int): Number of filters for this module

        Return:
            tf.Tensor: A () tensor output from this module

        """
        x = Conv2D(n_filters, (3,3), padding=self.padding)(inputs)
        x = LeakyReLU(alpha=self.leakyAlpha)(x)
        x = InstanceNormalization()(x)
        
        x = Conv2D(n_filters, (1,1), padding=self.padding)(x)
        x = LeakyReLU(alpha=self.leakyAlpha)(x)
        x = InstanceNormalization()(x)
        
        return x
    
    def segmentationLayer(self, inputs, n_filters):
        """ Improved UNET Segmentation Layer
        
        ""

        Parameters:
            inputs (tf.Tensor): A () tensor inputted into the layer 
            n_filters (int): Number of filters for this module

        Return:
            tf.Tensor: A () tensor output from this module

        """
        x = Conv2D(n_filters, (1,1), padding=self.padding)(inputs)
        x = LeakyReLU(alpha=self.leakyAlpha)(x)
        
        return x
        
    def encoderBlock(self, inputs, n_filters, strides=(1,1)):
        """ Improved UNET Encoder Block
        
        1 convolution layer and 1 context module

        Parameters:
            inputs (tf.Tensor): A () tensor inputted into the block 
            n_filters (int): Number of filters for this block
            strides ((int, int)): Strides for the convolution

        Return:
            tf.Tensor: A () tensor output from this module

        """
    
        x = Conv2D(n_filters, (3,3), strides=strides, padding=self.padding)(inputs)
        convOutput = LeakyReLU(alpha=self.leakyAlpha)(x)
        contextOutput = self.contextModule(convOutput, n_filters)
        x = Add()([convOutput, contextOutput])

        return x

    def decoderBlock(self, prev_layer_input, skip_layer_input, n_filters):
        """ Improved UNET Decoder Block
        
        1 concatenation layer
        1 localisation layer
        1 upsampling layer

        Parameters:
            prev_layer_input (tf.Tensor): A () tensor input from the previous layer
            skip_layer_input (tf.Tensor): A () tensor from the skip connection
            n_filters (int): Number of filters for this block

        Return:
            (tf.Tensor,tf.Tensor): A () tensor from the localisation layer 
                and a () tensor output from this module

        """
        x = Concatenate()([prev_layer_input, skip_layer_input])
        localisation = self.localisationModule(x, n_filters)
        x = self.upsamplingModule(localisation, n_filters / 2)
        
        return localisation, x
        
    def modelArchitecture(self):
        """ Defines Improved UNET model network
        Described in the paper Brain Tumor Segmentation and Radiomics
        

        Return:
            tf.keras.models.Model: CNN defined by Improved UNET architecture

        """
        
        inputs = Input(shape=(self.img_height,self.img_width,1))

        # Stack 5 encoders, doubling number of filter in each block.
        # Saving a skip connection for each
        encoder = self.encoderBlock(inputs, self.init_filters)
        skip1 = encoder
        encoder = self.encoderBlock(encoder, self.init_filters * 2, strides=(2,2))
        skip2 = encoder
        encoder = self.encoderBlock(encoder, self.init_filters * 4, strides=(2,2))
        skip3 = encoder
        encoder = self.encoderBlock(encoder, self.init_filters * 8, strides=(2,2))
        skip4 = encoder        
        encoder = self.encoderBlock(encoder, self.init_filters * 16, strides=(2,2))
        
        # Upsampling
        upsample = self.upsamplingModule(encoder, self.init_filters * 8)
        
        # Stack 3 decoders, halving number of filter in each block.
        # Saving localisation layers
        x, decoder = self.decoderBlock(upsample, skip4, self.init_filters * 8)
        localisation1, decoder = self.decoderBlock(decoder, skip3, self.init_filters * 4)
        localisation2, decoder = self.decoderBlock(decoder, skip2, self.init_filters * 2)
        
        # Upsample the first segmentation layer, Add the segmentation layers
        segmentation1 = self.segmentationLayer(localisation1, 1)
        upScaleSegmentation1 = UpSampling2D(interpolation='bilinear')(segmentation1)
        segmentation2 = self.segmentationLayer(localisation2, 1)
        firstSum = Add()([upScaleSegmentation1, segmentation2])
        
        # Final decoding block
        # Concatenation, Convolution, Segmentation
        decoder = Concatenate()([decoder, skip1])
        decoder = Conv2D(self.init_filters * 2, (3,3), padding=self.padding)(decoder)
        decoder = LeakyReLU(alpha=self.leakyAlpha)(decoder)
        segmentation3 = self.segmentationLayer(decoder, 1)
        
        # Upsample the first segmentation sum, Add the final segmentation layers
        upScaleFirstSum = UpSampling2D(interpolation='bilinear')(firstSum)
        secondSum = Add()([upScaleFirstSum, segmentation3])
        
        outputs = Activation('sigmoid')(secondSum)
        
        model = Model(inputs, outputs)
        
        return model