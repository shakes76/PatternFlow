#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all the necessary libraries

import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
get_ipython().run_line_magic('matplotlib', 'inline')

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
from keras import backend as K

from tqdm import tqdm_notebook, tnrange

from itertools import chain

from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label

from sklearn.model_selection import train_test_split


# In[2]:


# check the tf version
print("Tensorflow Version: ", tf.keras.__version__)


# In[3]:


# loading the dataset 
isic_features = next(os.walk("D:/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2"))[2] # returns all the files 
isic_labels = next(os.walk("D:/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2"))[2] # returns all the files

print("Number of images in features folder = ", len(isic_labels))
print("Number of images in labels folder = ", len(isic_labels))


# In[4]:


# sorting of data with respect to labels
isic_features_sort = sorted(isic_features) 
isic_labels_sort = sorted(isic_labels)


# In[5]:


# initialising the compression dimensions
img_width = 256
img_height = 256
border = 5


# In[6]:


# get and resize the training input dataset
def load_features(inp_path, ids):
    """ This function loads the data from training input folder into grayscale mode 
    and normalizes the features to 0 and 1 """
    X = np.zeros((len(ids), img_height, img_width, 1), dtype = np.float32)
    for n, id_ in tqdm_notebook(enumerate(ids), total = len(ids)): # capture all the images ids using tqdm       
        # load the image
        img = load_img(inp_path + id_, color_mode = 'grayscale') 
        x_img = img_to_array(img) # convert images to array
        x_img = resize(x_img, (256,256,1), mode = 'constant', preserve_range = True)
        X[n] = x_img/255 # normalize the images
    return X    


# In[9]:


# function for loading the training groundtruth dataset
def load_labels(inp_path, ids):
    """ This function loads the data from training groundtruth folder into grayscale mode """
    X = np.zeros((len(ids), img_height, img_width,1), dtype = np.uint8)
    for n, id_ in tqdm_notebook(enumerate(ids), total = len(ids)): # capture all the images ids using tqdm 
        # load the image
        img = load_img(inp_path + id_,color_mode = 'grayscale') 
        x_img = img_to_array(img) # convert images to array
        x_img = resize(x_img,(256,256,1),mode = 'constant', preserve_range = True)
        X[n] = x_img
    return X


# In[10]:


# Loading the images for the training input data set
X_isic_train = load_features("D:/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1-2_Training_Input_x2/", isic_features_sort)


# In[11]:


# Loading the images for the training groundtruth data set
y_isic_train=load_labels("D:/ISIC2018_Task1-2_Training_Data/ISIC2018_Task1_Training_GroundTruth_x2/", isic_labels_sort)


# In[12]:


# train-validation-test split
X_train, X_test, y_train, y_test = train_test_split(X_isic_train, y_isic_train, test_size = 0.20, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.25, random_state = 42)


# In[13]:


# normalizing the labels to 0 and 1 
y_train_sc = y_train//255
y_test_sc = y_test//255
y_val_sc = y_val//255


# In[14]:


# one hot encoding 
y_train_encode = to_categorical(y_train_sc) 
y_test_encode = to_categorical(y_test_sc) 
y_val_encode = to_categorical(y_val_sc) 


# In[15]:


def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    """ This function is used to add 2 convolutional layers with the parameters passed to it"""
    
    # first layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = "he_normal", padding = "same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x) # it draws samples from a truncated normal distribution centered on 0 to normalize the outputs from previous layers
    x = Activation("relu")(x)
    
    # second layer
    x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size), kernel_initializer = "he_normal", padding = "same")(input_tensor)
    if batchnorm:
            x = BatchNormalization()(x) # it draws samples from a truncated normal distribution centered on 0 to normalize the outputs from previous layers
    x = Activation("relu")(x)    
    return x


# In[16]:


def get_unet(input_img, n_filters = 16, dropout = 0.1, batchnorm = True):
    """ This function is used to generate and define the U-Net architecture - Encoder and Decoder"""
    
    # contracting path 
    c1 = conv2d_block(input_img, n_filters = n_filters*1, kernel_size = 3, batchnorm = batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters = n_filters*2, kernel_size = 3, batchnorm = batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = conv2d_block(p2, n_filters = n_filters*4, kernel_size = 3, batchnorm = batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = conv2d_block(p3, n_filters = n_filters*8, kernel_size = 3, batchnorm = batchnorm)
    p4 = MaxPooling2D((2, 2)) (c4)
    p4 = Dropout(dropout)(p4)
    
    c5 = conv2d_block(p4, n_filters = n_filters*16, kernel_size = 3, batchnorm = batchnorm)
    
    # expansive path 
    u6 = Conv2DTranspose(n_filters*8, (3, 3), strides = (2, 2), padding = 'same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = conv2d_block(u6, n_filters = n_filters*8, kernel_size = 3, batchnorm = batchnorm)

    u7 = Conv2DTranspose(n_filters*4, (3, 3), strides = (2, 2), padding = 'same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = conv2d_block(u7, n_filters = n_filters*4, kernel_size = 3, batchnorm = batchnorm)

    u8 = Conv2DTranspose(n_filters*2, (3, 3), strides = (2, 2), padding = 'same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = conv2d_block(u8, n_filters = n_filters*2, kernel_size = 3, batchnorm = batchnorm)

    u9 = Conv2DTranspose(n_filters*1, (3, 3), strides = (2, 2), padding = 'same') (c8)
    u9 = concatenate([u9, c1])
    u9 = Dropout(dropout)(u9)
    c9 = conv2d_block(u9, n_filters = n_filters*1, kernel_size = 3, batchnorm = batchnorm)
    
    outputs = Conv2D(2, (1, 1), activation = 'sigmoid') (c9)
    model = Model(inputs = [input_img], outputs = [outputs])
    return model


# In[17]:


# dice coeffient
def dice_coeffient(y_true, y_pred, smooth = 1):
    """ this function is used to gauge the similarity of two samples """
    intersect = K.sum(K.abs(y_true * y_pred), axis = [1,2,3])
    union = K.sum(y_true,[1,2,3]) + K.sum(y_pred,[1,2,3]) - intersect
    coeff_dice = K.mean((intersect + smooth) / (union + smooth), axis = 0)
    return coeff_dice


# In[18]:


# dice loss function
def dice_loss(y_true, y_pred, smooth = 1):
    return 1 - dice_coeffient(y_true, y_pred, smooth = 1)


# In[20]:


input_img = Input((img_height, img_width, 1), name = 'img')
model = get_unet(input_img, n_filters = 16, dropout = 0.05, batchnorm = True)
# compiling the model with Adam optimizer and dice loss
model.compile(optimizer = Adam(), loss = dice_loss, metrics = ["accuracy",dice_coeffient])


# In[21]:


model.summary()


# In[22]:


# A callback is a set of functions to be applied at given stages of the training procedure.
callbacks = [
    EarlyStopping(patience = 10, verbose = 1),
    ReduceLROnPlateau(factor = 0.1, patience = 5, min_lr = 0.00001, verbose = 1),
    ModelCheckpoint('ISIC_model.h5', verbose = 1, save_best_only = True, save_weights_only = True)
]


# In[23]:


results = model.fit(X_train, y_train_encode, batch_size = 32, epochs = 60, callbacks = callbacks, validation_data = (X_val, y_val_encode))


# In[24]:


# plot for training loss and validation loss wrt epochs
plt.figure(figsize = (8, 8))
plt.title("dice loss")
plt.plot(results.history["loss"], label = "training_loss")
plt.plot(results.history["val_loss"], label = "validation_loss")
plt.plot( np.argmin(results.history["val_loss"]), np.min(results.history["val_loss"]), marker = "x", color = "r", label = "best model")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend();


# In[25]:


# plot for training accuracy and validation accuracy wrt epochs
plt.figure(figsize = (8,8))
plt.title("Classification Accuracy")
plt.plot(results.history["accuracy"],label = "training_accuracy")
plt.plot(results.history["val_accuracy"],label = "validation_accuracy")
plt.plot(np.argmin(results.history["val_accuracy"]),np.max(results.history["val_accuracy"]),marker = "x",color = "r",label = "best model")
plt.xlabel("Epochs")
plt.legend();


# In[26]:


# load best model
model.load_weights('ISIC_model.h5')


# In[30]:


# predict on model
test_preds = model.predict(X_test,verbose = 1) 
test_preds_max = np.argmax(test_preds,axis = -1)
n,h,w,g = y_test.shape
test_preds_reshape = test_preds_max.reshape(n,h,w,g)


# In[31]:


def plot_ISIS(X, y, Y_pred,ix = None):
    """ this function returns the input image, true image and predicted image"""
    if ix is None:
        ix = random.randint(0, len(X))
    else:
        ix = ix    

    fig, ax = plt.subplots(1, 3, figsize = (20, 10))
    
    ax[0].imshow(X[ix, ..., 0], cmap = 'gray')
    ax[0].contour(X[ix].squeeze(), colors = 'k', levels = [0.5])
    ax[0].set_title('Input Image')
        
    ax[1].imshow(y[ix, ..., 0], cmap = 'gray')
    ax[1].contour(y[ix].squeeze(), colors = 'k', levels = [0.5])
    ax[1].set_title('True Image')
    
    ax[2].imshow(Y_pred[ix, ..., 0], cmap='gray')
    ax[2].contour(Y_pred[ix].squeeze(), colors = 'k', levels = [0.5])
    ax[2].set_title('Predicted Image')


# In[36]:


# Check if testing data looks all right
plot_ISIS(X_test,y_test,test_preds_reshape)


# In[ ]:




