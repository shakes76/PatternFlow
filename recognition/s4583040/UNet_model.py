"""
Improved UNet model made for COMP3710 report

Author: Siwan Li
Student ID: s4583040
Date: 31 October 2021
GitHub Name: Kirbologist
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    # Dice coefficient, but with an added smooth factor
    numerator = K.sum(y_true * y_pred, axis=[1,2,3])
    denominator = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
    return K.mean( (2. * numerator + smooth) / (denominator + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def Context_module(filters, input_layer, layer_name):
    # Module for computing activations in the context pathway, as given by the original paper
    module = layers.Conv2D(filters, kernel_size =3, padding = 'same')(input_layer)
    module = layers.BatchNormalization()(module)
    module = layers.LeakyReLU(alpha=0.01)(module)
    module = layers.Dropout(.3)(module)
    module = layers.Conv2D(filters, kernel_size =3, padding = 'same')(module)
    module = layers.BatchNormalization()(module)
    module = layers.LeakyReLU(alpha=0.01)(module)

    return module
    
def Upsampling_module(filters, input_layer):
    # Combines the upsampling layers into one module, following the structure given by the original paper
    module = layers.UpSampling2D(size=(2,2))(input_layer)
    module = layers.Conv2D(filters, kernel_size =3, padding = 'same')(module)
    module = layers.BatchNormalization()(module)
    module = layers.LeakyReLU(alpha=0.01)(module)
    
    return module
    
def Localisation_module(filters, input_layer):
    # Module used in localisation pathway, as given by the original paper
    module = layers.Conv2D(filters, kernel_size =3, padding = 'same')(input_layer)
    module = layers.BatchNormalization()(module)
    module = layers.LeakyReLU(alpha=0.01)(module)
    module = layers.Dropout(.3)(module)
    module = layers.Conv2D(filters, kernel_size =1, padding = 'same')(module)
    module = layers.BatchNormalization()(module)
    module = layers.LeakyReLU(alpha=0.01)(module)
    
    return module

def Improved_UNet_Model():
    filters = 16
    input_layer = Input((256,256,3))

    # Downsampling
    layer1 = layers.Conv2D(filters, kernel_size =3, padding = 'same')(input_layer)
    layer1 = layers.LeakyReLU(alpha=0.01)(layer1)
    layer1_module = Context_module(filters, layer1, "context_module1")
    output1 = layers.Add()([layer1, layer1_module])
    
    layer2 = layers.Conv2D(filters*2, kernel_size =3, strides = 2, padding = 'same')(output1)
    layer2 = layers.LeakyReLU(alpha=0.01)(layer2)
    layer2_module = Context_module(filters*2, layer2, "context_module2")
    output2 = layers.Add()([layer2, layer2_module])
    
    layer3 = layers.Conv2D(filters*4, kernel_size =3, strides = 2, padding = 'same')(output2)
    layer3 = layers.LeakyReLU(alpha=0.01)(layer3)
    layer3_module = Context_module(filters*4, layer3, "context_module3")
    output3 = layers.Add()([layer3, layer3_module])
    
    layer4 = layers.Conv2D(filters*8, kernel_size =3, strides = 2, padding = 'same')(output3)
    layer4 = layers.LeakyReLU(alpha=0.01)(layer4)
    layer4_module = Context_module(filters*8, layer4, "context_module4")
    output4 = layers.Add()([layer4, layer4_module])
    
    layer5 = layers.Conv2D(filters*16, kernel_size =3, strides = 2, padding = 'same')(output4)
    layer5 = layers.LeakyReLU(alpha=0.01)(layer5)
    layer5_module = Context_module(filters*16, layer5, "context_module5")
    output5 = layers.Add()([layer5, layer5_module])
    
    
    # Upsampling
    layer6 = Upsampling_module(filters*8, output5)
    output6 = layers.concatenate([output4, layer6])
    
    layer7 = Localisation_module(filters*8, output6)
    layer7_module = Upsampling_module(filters*4, layer7)
    output7 = layers.concatenate([output3, layer7_module])
    
    layer8 = Localisation_module(filters*4, output7)
    layer8_module = Upsampling_module(filters*2, layer8)
    output8 = layers.concatenate([output2, layer8_module])
    
    layer9 = Localisation_module(filters*2, output8)
    layer9_module = Upsampling_module(filters, layer9)
    output9 = layers.concatenate([output1, layer9_module])
    
    segmentation_1 = layers.Conv2D(1, kernel_size =3, padding = 'same')(layer7)
    segmentation_1 = layers.UpSampling2D(size=(8,8))(segmentation_1)
    segmentation_2 = layers.Conv2D(1, kernel_size =3, padding = 'same')(layer8)
    segmentation_2 = layers.UpSampling2D(size=(4,4))(segmentation_2)
    final_layer_output = layers.Conv2D(1, kernel_size =3, padding = 'same')(output9)
    
    output = layers.Add()([segmentation_1, segmentation_2, final_layer_output])
    output = layers.Activation('sigmoid')(output)
    
    model = Model(input_layer, output, name="improved_unet_model")
    model.summary()
    
    return model
