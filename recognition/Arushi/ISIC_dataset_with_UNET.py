#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Import all the Libraries
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical


# In[4]:


# check the version
print("Tensorflow Version: ", tf.keras.__version__)


# In[5]:


# Initialising the compression dimensions
img_width = 256
img_height =256
border = 5


# In[6]:


# Loading the dataset 
isic_features = next(os.walk("D:/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2"))[2] # returns all the files "DIR."
isic_labels = next(os.walk("D:/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2"))[2] # returns all the files "DIR."

print("Number of images in features folder = ", len(isic_labels))
print("Number of images in labels folder = ", len(isic_labels))


# In[7]:


isic_features_sort = sorted(isic_features) # Sorting of data wit respect to labels
isic_labels_sort = sorted(isic_labels) # Sorting of data wit respect to labels


# In[8]:


def load_features(inp_path,ids):
    X= np.zeros((len(ids),img_height,img_width,1),dtype=np.float32)
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)): # capture all the images ids using tqdm       
        img = load_img(inp_path+id_, color_mode = 'grayscale')  
        x_img = img_to_array(img) # Convert images to array
        x_img = resize(x_img,(256,256,1),mode = 'constant',preserve_range = True)
        X[n] = x_img/255 # Normalize the images
    return X    


# In[9]:


def load_labels(inp_path,ids):
    X= np.zeros((len(ids),img_height,img_width,1),dtype=np.uint8)
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        img = load_img(inp_path+id_,color_mode = 'grayscale') # Load images here
        x_img = img_to_array(img) # Convert images to array
        x_img = resize(x_img,(256,256,1),mode = 'constant', preserve_range = True)
        X[n] = x_img
    return X


# In[10]:


# Loading the images for the training input data set
X_isic_train = load_features("D:/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2/",isic_features_sort)


# In[11]:


# Loading the images for the training groundtruth data set
y_isic_train=load_labels("D:/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2/",isic_labels_sort)


# In[12]:


# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_isic_train, y_isic_train, test_size = 0.20, random_state = 42)


# In[13]:


# train-val-test split
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)


# In[14]:


y_train_sc = y_train//255
y_test_sc = y_test//255
y_val_sc = y_val//255


# In[15]:


# one hot encoding
y_train_encode = to_categorical(y_train_sc) 
y_test_encode = to_categorical(y_test_sc) 
y_val_encode = to_categorical(y_val_sc) 


# In[16]:


# Dice Coeffient
from keras import backend as K
def dice_coeff(y_true, y_pred, smooth=1):
    intersect = K.sum(K.abs(y_true * y_pred), axis=[1,2,3])
    union = K.sum(y_true,[1,2,3])+K.sum(y_pred,[1,2,3])-intersect
    coeff_dice = K.mean((intersect + smooth) / (union + smooth), axis=0)
    return coeff_dice


# In[17]:


def dice_loss(y_true, y_pred, smooth = 1):
    return 1 - dice_coeff(y_true, y_pred, smooth = 1)


# In[18]:


def conv2d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    layer = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    # It draws samples from a truncated normal distribution centered on 0 
    if batchnorm:
        layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    # second layer
    layer = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    return layer


# In[19]:


def get_unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):
    # contracting path - reduce enoder part
    c1 = conv2d_block(input_img, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path - Decoder part
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = Conv2D(2, (1, 1), activation='sigmoid') (c9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model


# In[20]:


input_img = Input((img_height, img_width, 1), name='img')
model = get_unet(input_img, n_filters=16, dropout=0.05, batchnorm=True)

model.compile(optimizer=Adam(), loss=dice_loss, metrics=["accuracy",dice_coeff])


# In[21]:


model.summary()


# In[ ]:




