#from PIL import Image, ImageOps 
from PIL import ImageOps
import image
import numpy as np
from matplotlib import pyplot as plt
import os
import tqdm
from tqdm import tqdm_notebook, tnrange
from skimage.transform import resize

import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D, GlobalMaxPool2D
from tensorflow.keras.layers import concatenate, add, LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


#############################################################################################################################################################################################################

imgName_X_train = next(os.walk("C:/Users/s4586360/Downloads/keras_png_slices_data/keras_png_slices_train"))[2] # list of names all images in the given path
imgName_y_train = next(os.walk("C:/Users/s4586360/Downloads/keras_png_slices_data/keras_png_slices_seg_train"))[2] # list of names all images in the given path
print("No. of training images = ", len(imgName_X_train))
print("No. of training images labels = ", len(imgName_y_train))

print ("")

imgName_X_validate = next(os.walk("C:/Users/s4586360/Downloads/keras_png_slices_data/keras_png_slices_validate"))[2] # list of names all images in the given path
imgName_y_validate = next(os.walk("C:/Users/s4586360/Downloads/keras_png_slices_data/keras_png_slices_seg_validate"))[2] # list of names all images in the given path
print("No. of validating images = ", len(imgName_X_validate))
print("No. of validating images labels = ", len(imgName_y_validate))

#print ("")

#imgName_X_test = next(os.walk("C:/Users/s4586360/Downloads/keras_png_slices_data/keras_png_slices_test"))[2] # list of names all images in the given path
#imgName_y_test = next(os.walk("C:/Users/s4586360/Downloads/keras_png_slices_data/keras_png_slices_seg_test"))[2] # list of names all images in the given path
#print("No. of testing images = ", len(imgName_X_test))
#print("No. of testing images labels = ", len(imgName_y_test))

#############################################################################################################################################################################################################

X_train = np.zeros((len(imgName_X_train), 256, 256, 1), dtype=np.float32)
y_train = np.zeros((len(imgName_y_train), 256, 256, 1), dtype=np.float32)

X_validate = np.zeros((len(imgName_X_validate), 256, 256, 1), dtype=np.float32)
y_validate = np.zeros((len(imgName_y_validate), 256, 256, 1), dtype=np.float32)

X_test = np.zeros((len(imgName_X_test), 256, 256, 1), dtype=np.float32)
y_test = np.zeros((len(imgName_y_test), 256, 256, 1), dtype=np.float32)

#############################################################################################################################################################################################################
############################ Generator ###########################

def conv2d_func(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """Function to add 2 convolutional layers with the parameters passed to it"""
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
              kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    return x

def generator(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """Function to define the UNET Model"""
    # Contracting Path
    c1 = conv2d_func(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2))(c1)
    p1 = Dropout(dropout)(p1)
    
    c2 = conv2d_func(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2))(c2)
    p2 = Dropout(dropout)(p2)
    
    c3 = conv2d_func(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2))(c3)
    p3 = Dropout(dropout)(p3)
    
    c4 = conv2d_func(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2))(c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_func(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)
    
    # Expansive Path
    u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_func(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
    
    u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_func(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
    
    u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_func(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
    
    u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_func(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
	

#############################################################################################################################################################################################################

########################### Discriminator ###########################

def downsample(filters, size, apply_batchnorm=True):
  init = tf.random_normal_initializer(0., 0.02)

  conv = Conv2D(filters=filters, kernel_size=(size,size) , strides=2, padding='same', kernel_initializer=init)

  if apply_batchnorm:
    conv = BatchNormalization()(conv)

  conv = LeakyReLU(alpha=0.1)(conv)

  return conv
  
def discriminator(inp_img, gen_img):

  init = tf.random_normal_initializer(0., 0.02)

  inp_img_d = Input(shape=inp_img)
  gen_img_d = Input(shape=gen_img)

  concat_img = concatenate([inp_img_d, gen_img_d])

  d1 = downsample(64, 2, False)(concat_img)
  d2 = downsample(128, 2, False)(d1)
  d3 = downsample(256, 2, False)(d2)
  d4 = downsample(512, 2, False)(d3)

  d_final = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d4)
  disc_out = Activation('sigmoid')(d_final)  
  
  disc_model = Model([inp_img_d, gen_img_d], disc_out)
  
  return disc_model

#############################################################################################################################################################################################################

############# Model training #########################




