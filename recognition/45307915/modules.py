import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, Concatenate, Dropout
from tensorflow.keras.models import Model

class ImprovedUNETModel():
    
    def __init__(self, n_classes, img_width=IMAGE_WIDTH, img_height=IMAGE_HEIGHT):
        """ Create a new Improved UNET Model instance.
        
        Described in the paper Brain Tumor Segmentation and Radiomics
        
        Parameters:
            n_classes (int): Number of classes
            img_wdith (int): Image Width
            img_height (int): Image Height
        """
        self.n_classes = n_classes
        self.img_width = img_width
        self.img_height = img_height
        
        self.padding = 'same'
        self.leakyAlpha = 1e-2
        self.dropoutRate = 0.3
        
        self.model = self.modelArchitecture()
        
    def contextBlock(self, inputs, n_filters):
        """ Improved UNET Context Block
        
        "Each context module is in fact a pre-activation residual block with two
        3x3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between"

        Parameters:
            inputs ():
            n_filters (int):

        Return:
            int: something

        """
        x = BatchNormalization()(inputs)
        x = LeakyReLU(alpha=self.leakyAlpha)(x)
        x = Conv2D(n_filters, (3,3), padding=self.padding)(x)
        
        x = Dropout(self.dropoutRate)(x)
        
        x = BatchNormalization()(inputs)
        x = LeakyReLU(alpha=self.leakyAlpha)(x)
        x = Conv2D(n_filters, (3,3), padding=self.padding)(x)
        
        return x
        
    def encoderBlock(self, inputs, n_filters, max_pooling=True, strides=(1,1)):
        """ Improved UNET Encoder Block
        
        2 convolution layers and 1 Max Pooling layer
        Skip connection info is also saved

        Parameters:
            inputs ():
            n_filters (int):
            max_pooling (bool):

        Return:
            (,): something

        """
    
        x = Conv2D(n_filters, (3,3), strides=strides, padding=self.padding)(inputs)
        x = LeakyReLU(alpha=self.leakyAlpha)(x)
        x = self.contextModule(x, n_filters)
        
        
        if max_pooling:
            next_layer = MaxPooling2D(pool_size=(2,2))(x)
        else:
            next_layer = x

        skip_connection = x

        return next_layer, skip_connection

    # Mini decoder block of 1 up convolution layer, 1 concatenation layer (skip layer and up layer), and 2 convolution layer
    def decoderBlock(self, prev_layer_input, skip_layer_input, n_filters):
        """ Create a new Yolo Model instance.

        Parameters:
            

        Return:
            int: something

        """
    
        up = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(prev_layer_input)

        merge = Concatenate(axis=3)([up, skip_layer_input])

        x = Conv2D(n_filters, (3,3), activation='relu', padding='same')(merge)
        x = Conv2D(n_filters, (3,3), activation='relu', padding='same')(x)
        return x
    
        
    def modelArchitecture(self):
        """ Defines Improved UNET model network
        Described in the paper Brain Tumor Segmentation and Radiomics
        

        Return:
            tf.keras.models.Model: CNN defined by Improved UNET architecture

        """
        
        inputs = Input(shape=(256,256,1))

#         # Stack mini encoders, doubling number of filter in each block
#         encoder, skip1 = EncoderBlock(inputs, 32)
#         encoder, skip2 = EncoderBlock(encoder, 64)
#         encoder, skip3 = EncoderBlock(encoder, 128)
#         # encoder, skip4 = EncoderBlock(encoder, 256)

#         encoder, skip5 = EncoderBlock(encoder, 256, False)

#         # Stack mini decoders, halving number of filter in each block
#         # decoder = DecoderBlock(encoder, skip4, 256)
#         decoder = DecoderBlock(encoder, skip3, 128)
#         decoder = DecoderBlock(decoder, skip2, 64)
#         decoder = DecoderBlock(decoder, skip1, 32)

#         outputs = Conv2D(4, 1, activation='softmax')(decoder)

        
        
        model = Model(inputs, outputs)
        
        return model
    
    def compileModel(self): 
        self.model.compile(optimizer='adam',
                          loss=,
                          metrics=[tf.keras.metrics.IoU(num_classes=self.n_classes, target_class_ids=[0])])