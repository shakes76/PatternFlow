"""
COMP3710 Report

@author Linyuzhuo Zhou, 45545584
"""
# coding: utf-8



import os
import skimage.io as io
import matplotlib.pyplot as plt
import random
from PIL import Image

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from improved_unet import *
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow_addons.layers import InstanceNormalization
from keras.models import Model, model_from_json
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import load_model




import cv2
import glob


#### Load data



images = []
for img in glob.glob("ISIC2018_Task1-2_Training_Data\ISIC2018_Task1-2_Training_Input_x2/*.jpg"):
    n= cv2.imread(img,1)
    images.append(n)


masks = []
for img in glob.glob("ISIC2018_Task1-2_Training_Data\ISIC2018_Task1_Training_GroundTruth_x2/*.png"):
    m= cv2.imread(img,0)
    masks.append(m)


# Resize image to 256x256 and create two array for preparation of dataset spliting.



image_size = (256,256)




for i in range(len(images)):
    images[i] = cv2.resize(images[i],image_size,interpolation = cv2.INTER_CUBIC)
    images[i] = images[i]/255
for i in range(len(masks)):
    masks[i] = cv2.resize(masks[i],image_size,interpolation = cv2.INTER_CUBIC)
    masks[i] = np.round(masks[i]/255,0)




X = np.zeros([2594, 256, 256, 3])
y = np.zeros([2594, 256, 256])





for i in range(len(images)):
    X[i] = images[i]
    
for i in range(len(masks)):
    y[i] = masks[i]
        
y = y[:, :, :, np.newaxis]


# plt show X and y get right data


plt.imshow(X[7])



plt.imshow(y[7])


# split into train, validatate and test




X_train, X_test, Y_train, Y_test = train_test_split(X, y, train_size=0.7, random_state=7)
X_test, X_val, Y_test, Y_val = train_test_split(X_train, Y_train, test_size=0.3, random_state=7)


# check that X_train X_test Y_val X_val successful spilt

print(X_train.shape)
print(X_test.shape)
print(Y_val.shape)
print(X_val.shape)


# dice coefficient function and dice coefficient loss function




def dice_coefficient(y_true, y_pred):
    intersection = tf.reduce_sum(tf.math.multiply(y_true, y_pred))
    dsc = 2*intersection / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    return dsc




def dsc_loss(y_true, y_pred):
    return 1.0 - dice_coefficient(y_true, y_pred)


#### Build Model

model = improved_unet(256, 256, 3)



model.summary()



#### Compile Model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dice_coefficient, 'accuracy'])

history = model.fit(x = X_train, y=Y_train, epochs=25, validation_data=(X_val, Y_val), verbose=1, batch_size = 16)


#### plot training history


plt.figure(figsize= (10,10))
plt.plot(history.history['accuracy'], label = 'train_acc')
plt.plot(history.history['val_accuracy'], label = 'val_acc', linestyle='dashed')
plt.plot(history.history['dice_coefficient'], label = 'dice_coefficient')
plt.plot(history.history['val_dice_coefficient'], label = 'val_dice_coefficient', linestyle='dashed')
plt.title('Training History')
plt.ylabel('Result')
plt.xlabel('Epoch')
plt.legend(loc="lower right")
plt.show()





#### Calculate Average Dice Similarity


dsc = list()
for i in range(1):
    dsc += history.history['dice_coefficient']
print ("Average DSC:", sum(dsc)/25)    


#### plot predictions


predictions = model.predict(X_test)
for i in range(len(predictions)):
    predictions[i]=np.round(predictions[i],0)



n = 10 
plt.figure(figsize=(20, 20))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(X_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.figure(figsize=(20, 20))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(Y_test[i],)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.figure(figsize=(20, 20))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(predictions[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)






