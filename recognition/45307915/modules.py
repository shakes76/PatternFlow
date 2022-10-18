import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense, Reshape, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model

class ImprovedUNETModel():
    
    def __init__(self, n_classes, n_boxes, n_cells, img_width, img_height):
        """ Create a new Yolo Model instance.
        
        Parameters:
            n_classes (int): 
            n_boxes (int): 
            n_cells (int): 
            img_wdith (int): 
            img_height (int): 
        """
        self.n_classes = n_classes
        self.n_boxes = n_boxes
        self.n_cells = n_cells
        self.img_width = img_width
        self.img_height = img_height
        
        self.model = self.modelArchitecture()
        
    # Mini encoder block of 2 convolution layers and 1 Max Pooling layer
    # Skip connection info is also saved
    def encoderBlock(inputs, n_filters, max_pooling=True):
        """ Create a new Yolo Model instance.

        Parameters:
            imgWidth (int):

        Return:
            int: something

        """
    
        x = Conv2D(n_filters, (3,3), activation='relu', padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Conv2D(n_filters, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        if max_pooling:
            next_layer = MaxPooling2D(pool_size=(2,2))(x)
        else:
            next_layer = x

        skip_connection = x

        return next_layer, skip_connection

    # Mini decoder block of 1 up convolution layer, 1 concatenation layer (skip layer and up layer), and 2 convolution layer
    def decoderBlock(prev_layer_input, skip_layer_input, n_filters):
        """ Create a new Yolo Model instance.

        Parameters:
            

        Return:
            int: something

        """
    
        up = Conv2DTranspose(n_filters, (3,3), strides=(2,2), padding='same')(prev_layer_input)

        merge = Concatenate(axis=3)([up, skip_layer_input])

        x = Conv2D(n_filters, (3,3), activation='relu', padding='same')(merge)
        x = BatchNormalization()(x)
        x = Conv2D(n_filters, (3,3), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        return x
    
        
    def modelArchitecture(self):
        """ Defines Improved UNET model network
        Described in the paper Brain Tumor Segmentation and Radiomics
        

        Return:
            tf.keras.models.Model: CNN defined by Improved UNET architecture

        """
        
        inputs = Input(shape=(256,256,1))

        # Stack mini encoders, doubling number of filter in each block
        encoder, skip1 = self.encoderBlock(inputs, 32)
        encoder, skip2 = self.encoderBlock(encoder, 64)
        encoder, skip3 = self.encoderBlock(encoder, 128)
        # encoder, skip4 = EncoderBlock(encoder, 256)

        encoder, skip5 = self.encoderBlock(encoder, 256, False)

        # Stack mini decoders, halving number of filter in each block
        # decoder = DecoderBlock(encoder, skip4, 256)
        decoder = self.decoderBlock(encoder, skip3, 128)
        decoder = self.decoderBlock(decoder, skip2, 64)
        decoder = self.decoderBlock(decoder, skip1, 32)

        outputs = Conv2D(4, 1, activation='softmax')(decoder)

        
        
        model = Model(inputs, outputs)
        
        return model
    
    def compileModel(self): 
        self.model.compile(optimizer='adam',
                          loss=,
                          metrics=[tf.keras.metrics.IoU(num_classes=self.n_classes, target_class_ids=[0])])