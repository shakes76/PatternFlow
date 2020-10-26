"""
COMP3710 Report 

@author Huizhen 
"""

#%%
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from random import shuffle
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose


print("TensorFlow Version: ", tf.__version__) 

# Define DSC loss function
def dsc(x, y):
    """
    Parameters:
        x , y (numpy.array): Prediction and Grouptruth images.

    Return:
        dsc (float): The DSC value.
    """
    TP = tf.reduce_sum(tf.math.multiply(x,y))
    FP_FN = tf.reduce_sum(tf.cast(tf.math.logical_xor(x,y),tf.float32))
    dsc = 2*TP / (2*TP + FP_FN)
    return dsc.numpy()

# Define Block Functions
def context_module(inputs, filters, dropout_rate):
    bn1 = BatchNormalization()(inputs)
    relu1 = Activation('relu')(bn1)
    conv1 = Conv2D(filters, (3,3), padding='same')(relu1)
    dropout = Dropout(dropout_rate)(conv1)
    bn2 = BatchNormalization()(dropout)
    relu2 = Activation('relu')(bn2)
    conv2 = Conv2D(filters, (3,3), padding='same')(relu2)
    output = Add()([inputs, conv2])
    return output

def localization_module(upsample, cm, filters):
    concat = concatenate([upsample, cm])
    conv1 = Conv2D(filters*2, (3,3), padding='same')(concat)
    conv2 = Conv2D(filters, (1,1))(conv1)
    return conv2

def upsampling_module(inputs, filters):
    up = UpSampling2D(size=(2,2))(inputs)
    conv = Conv2D(filters, (3,3), padding='same')(up)
    return conv

def segmentation_addup_module(inputs1, inputs2):
    seg1 = Conv2DTranspose(1, (2,2), strides=(2,2), padding='same')(inputs1)
    seg2 = Conv2DTranspose(1, (2,2), strides=(1,1), padding='same')(inputs2)
    addup = Add()([seg1, seg2])
    return addup


def improved_unet():
    
    inputs = Input((128,128,3))

    c1 = Conv2D(16, (3,3), padding='same')(inputs)
    cm1 = context_module(c1, 16, 0.3)
    
    c2 = Conv2D(32, (3,3), strides=2, padding='same')(cm1)
    cm2 = context_module(c2, 32, 0.3)
    
    c3 = Conv2D(64, (3,3), strides=2, padding='same')(cm2)
    cm3 = context_module(c3, 64, 0.3)
    
    c4 = Conv2D(128, (3,3), strides=2, padding='same')(cm3)
    cm4 = context_module(c4, 128, 0.3)
    
    c5 = Conv2D(256, (3,3), strides=2, padding='same')(cm4)
    cm5 = context_module(c5, 256, 0.3)
    
    u1 = upsampling_module(cm5, 128)
    
    local1 = localization_module(u1, cm4, 128)
    
    u2 = upsampling_module(local1, 64)
    
    local2 = localization_module(u2, cm3,  64)
        
    u3 = upsampling_module(local2, 32)
    
    local3 = localization_module(u3, cm2,  32)
    
    u4 = upsampling_module(local3, 16)
    
    concat = concatenate([u4, cm1])
    
    c6 = Conv2D(32, (3,3), padding='same')(concat)
        
    sum1 = segmentation_addup_module(local2, local3)
    sum2 = segmentation_addup_module(sum1, c6)
    
    outputs = Activation('sigmoid')(sum2)
    
    network = tf.keras.Model(inputs = [inputs], outputs = [outputs])
    return network

#model = improved_unet()
#model.summary()

def resize_image(image, h=128, w=128, divide255 = True):
    # /225 is neccessary for input images due to non-integer value after resize
    new_image = tf.image.resize(image, [h, w]).numpy()/(255 if divide255 else 1)  
    return new_image

def load_batch(input_images: list, output_images: list, input_folder: str, output_folder: str, batch_size=16):
    """
    input_images : image name list
    output_images : image name list
    input_folder : folder name
    output_folder : folder name
    """
    idx = 0
    i = []
    o = []
    while 1:
        while 1:
            i.append(resize_image(mpimg.imread(input_folder + '/' + input_images[idx])))
            o.append(resize_image(mpimg.imread(output_folder + '/' + output_images[idx])[:,:,np.newaxis], divide255 = False))
            idx += 1
            if len(i) == batch_size: 
                yield (np.array(i), np.array(o))
                i = []
                o = []
            if idx == len(input_files):
                idx = 0
                

# data path
input_folder = '/Users/taamsmac/Downloads/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2'
output_folder = '/Users/taamsmac/Downloads/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2'

# image files name
input_images = sorted([file for file in os.listdir(input_folder) if file.endswith('jpg')])
output_images = sorted([file for file in os.listdir(output_folder) if file.endswith('png')])
assert len(input_images)==len(output_images), 'input and output lists have different length'

# split into train, validatate and test
idx = list(range(len(input_images)))
shuffle(idx)
train_idx = idx[:1800]
validate_idx = idx[1800:2200]
test_idx = idx[2200:]
print(f'train, validate, test set have length: {len(train_idx)}, {len(validate_idx)}, {len(test_idx)}')

# train, test, validate images file name
train_input_images = [input_images[i] for i in train_idx]
train_output_images = [output_images[i] for i in train_idx]
validate_input_images = [input_images[i] for i in validate_idx]
validate_output_images = [output_images[i] for i in validate_idx]
test_input_images = [input_images[i] for i in test_idx]
test_output_images = [output_images[i] for i in test_idx]



#generator = load_batch(Input, Output)
#b1 = next(generator)


# parameters

# layers

# build networks

# build model

# trian

# test

# plot
# %%
