"""
Model architecture for improved UNet.

@author Haoxiang Zhang
@email haoxiang.zhang@uqconnect.edu.au
"""
import os
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.layers import MaxPooling2D

#UNET modules which are used in the improved UNET model
def main_mod(number_filter,inp,name_mod):
    #Construction: every two three*three convolution layers followed by a drop out layer with a probability of 0.3
    layer1=layers.Conv2D(number_filter,kernel_size=3,padding='same')(inp)
    layer1=layers.BatchNormalization()(layer1)
    layer1=layers.LeakyReLU(alpha=0.01)(layer1)
    layer1=layers.Dropout(0.3)(layer1)
    layer2=layers.Conv2D(number_filter,kernel_size=3,padding='same')(layer1)
    layer2=layers.BatchNormalization()(layer2)
    layer2=layers.LeakyReLU(alpha=0.01)(layer2)
    return layer2
    
def upsample_mod(number_filter,inp):
    #Construction: one upsampling layer and one convolution layer
    layer1=layers.UpSampling2D(size=(2,2))(inp)#kernel size to be 2*2
    layer2=layers.Conv2D(number_filter,kernel_size=3,padding='same')(layer1)
    layer2=layers.BatchNormalization()(layer2)
    layer2=layers.LeakyReLU(alpha=0.01)(layer2) 
    return layer2
    
def locate_mod(number_filter,inp):
    #Constuction: Two convolution layers, drop out layer and batch normalization layers
    layer1=layers.Conv2D(number_filter,kernel_size=3,padding='same')(inp)
    layer1=layers.BatchNormalization()(layer1)
    layer1=layers.LeakyReLU(alpha=0.01)(layer1)
    layer1=layers.Dropout(0.3)(layer1)
    layer2=layers.Conv2D(number_filter,kernel_size=1,padding='same')(layer1)
    layer2=layers.BatchNormalization()(layer2)
    layer2=layers.LeakyReLU(alpha=0.01)(layer2)
    return layer2

#Main part: improved UNET model
def Improved_UNet_model():
    '''
    Creating the encoding and decoding layers for the improved UNET
    '''
    number_filter=16
    layer_in=Input((256,256,3))
    #First Block: 16 
    first_block_layer1=layers.Conv2D(number_filter,kernel_size=3,padding='same')(layer_in)
    first_block_layer1=layers.LeakyReLU(alpha=0.01)(first_block_layer1)
    first_block_layer2=main_mod(number_filter,first_block_layer1,"main_mod1")
    first_block_output=layers.Add()([first_block_layer1,first_block_layer2])
    #Second Block: 32
    second_block_layer1=layers.Conv2D(number_filter*2,kernel_size=3,strides=2,padding='same')(first_block_output)
    second_block_layer1=layers.LeakyReLU(alpha=0.01)(second_block_layer1)
    second_block_layer2 = main_mod(number_filter*2, second_block_layer1, "main_mod2")
    second_block_output = layers.Add()([second_block_layer1, second_block_layer2])
    #Third Block: 64
    third_block_layer1=layers.Conv2D(number_filter*4,kernel_size=3,strides=2,padding='same')(second_block_output)
    third_block_layer1=layers.LeakyReLU(alpha=0.01)(third_block_layer1)
    third_block_layer2=main_mod(number_filter*4,third_block_layer1,"main_mod3")
    third_block_output=layers.Add()([third_block_layer1,third_block_layer2])  
    #Fourth Block: 128
    fourth_block_layer1=layers.Conv2D(number_filter*8,kernel_size=3,strides=2,padding='same')(third_block_output)
    fourth_block_layer1=layers.LeakyReLU(alpha=0.01)(fourth_block_layer1)
    fourth_block_layer2=main_mod(number_filter*8, fourth_block_layer1,"main_mod4")
    fourth_block_output=layers.Add()([fourth_block_layer1, fourth_block_layer2])
    
    #Fifth Block: 256
    fifth_block_layer1=layers.Conv2D(number_filter*16, kernel_size=3, strides=2, padding='same')(fourth_block_output)
    fifth_block_layer1=layers.LeakyReLU(alpha=0.01)(fifth_block_layer1) 
    fifth_block_layer2=main_mod(number_filter*16, fifth_block_layer1, "main_mod5")
    fifth_block_output=layers.Add()([fifth_block_layer1, fifth_block_layer2])
    
    #UpBlock six: 128
    six_block_layer1=upsample_mod(number_filter*8, fifth_block_output)
    #Connect with concatenate()
    six_block_output=layers.concatenate([fourth_block_output, six_block_layer1])
    
    #UpBlock seven: 128
    seven_block_layer1=locate_mod(number_filter*8, six_block_output)
    seven_block_layer2=upsample_mod(number_filter*4, seven_block_layer1)
    #Connect with concatenate()
    seven_block_output=layers.concatenate([third_block_output, seven_block_layer2])
    
    #UpBlock Eight: 64
    eight_block_layer1=locate_mod(number_filter*4, seven_block_output)
    eight_block_layer2=upsample_mod(number_filter*2, eight_block_layer1)
    #Connect with concatenate()
    eight_block_output=layers.concatenate([second_block_output, eight_block_layer2])
    
    #UpBlock Nine: 32
    nine_block_layer1=locate_mod(number_filter*2, eight_block_output)
    nine_block_layer2=upsample_mod(number_filter, nine_block_layer1)
    #Connect with concatenate()
    nine_block_output=layers.concatenate([first_block_output, nine_block_layer2])
    
    #Final segmentations
    seg1=layers.Conv2D(1, kernel_size =3, padding = 'same')(seven_block_layer1)
    seg1=layers.UpSampling2D(size=(8,8))(seg1)
    seg2=layers.Conv2D(1, kernel_size =3, padding = 'same')(eight_block_layer1)
    seg2=layers.UpSampling2D(size=(4,4))(seg2)
    
    final_output=layers.Conv2D(1, kernel_size =3, padding = 'same')(nine_block_output)
    output=layers.Add()([seg1, seg2, final_output])
    output=layers.Activation('sigmoid')(output)
    improved_UNET_model=Model(layer_in, output, name="improved_unet_model")
    improved_UNET_model.summary()
    return improved_UNET_model
