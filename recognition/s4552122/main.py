"""
ISICs dataset by an improved Unet to make image segment.
ISICs data set concludes thousands of Skin Lesion images. 
This recognition algorithm aims to automatically do Lesion Segmentation through an improved unet model

@author Xiaoqi Zhuang
@email x.zhuang@uqconnect.edu.au
"""

#Import 
import tensorflow as tf
import os
from matplotlib import image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, Flatten, MaxPooling2D, BatchNormalization, Conv2DTranspose
from keras.optimizers import SGD, Adam
tf.random.Generator = None
import tensorflow_addons as tfa
import cv2
from sklearn.model_selection import train_test_split
from model.py import *


#loading features/original images
X = []
Path = "ISIC2018_Task1-2_Training_Input_x2"
for img in os.listdir(Path):
    im = image.imread("ISIC2018_Task1-2_Training_Input_x2/" + img) 
    X.append(im)

#Loading segmentation images
Y = []
Path = "ISIC2018_Task1_Training_GroundTruth_x2"
for img in os.listdir(Path):
    im = image.imread("ISIC2018_Task1_Training_GroundTruth_x2/" + img) 
    Y.append(im)

#Remeber the original images size for inversing them into the evluating step.
X_, X_test_real, y_, y_test_real = train_test_split(X, Y, test_size = 0.2, random_state = 33)
#Reducing unnessary variabls'smemory usage
del X_
del y_

#Data processing
X = np.array(X)
#Since images have different shapes, I resize them to 256*256 images.
for i in range(len(X)):
    X[i] = cv2.resize(X[i],(256,256))
#Reshape the image to be (number of images, 256, 256, 3)
images = np.zeros([2594,256,256,3])
for i in range(len(X)):
    images[i] = X[i]
images = images/255
del X

for i in range(len(Y)):
    Y[i] = cv2.resize(Y[i],(256,256))
#Reshape the image to be (number of images, 256, 256, 3)
Y = np.array(Y)[:,:,:,np.newaxis]
#Make the segmentation images' labels just be 0 and 1.
Y = np.around(Y)

#Split training set, validation set, testing set.
X_tv, X_test, y_tv, y_test = train_test_split(images, Y, test_size = 0.2, random_state = 33)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size = 0.25, random_state = 33)
print("X_train shape: ", X_train.shape)
print("X_val shape: ", X_val.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_val shape: ", y_val.shape)
print("y_test shape: ", y_test.shape)
del X_tv
del y_tv

#Building the improved uNet model
model = improvedUnet()
model.summary()
results = model.fit(X_train, y_train, validation_data= (X_val, y_val), batch_size=16, epochs=5)

#Prediction
preds_test = model.predict(X_test, verbose=1)
#Nomorlization the results into [0,1]
preds_test = np.around(preds_test)

#Evaluation: Dice similarity coefficient on the test set
def dice_similar_coef(y_true, y_pred):
    
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
#inverse the image size
preds_test = preds_test.reshape([519,256,256])
preds_test_real = []
for i in range(len(y_test_real)):
    preds_test_real.append(cv2.resize(preds_test[i], (y_test_real[i].shape[1],y_test_real[i].shape[0])))
#Calculate the average dice similarity coefficients
sum_Dice = 0
avg_Dice = 0
for i in range(len(y_test_real)):
    dice = dice_similar_coef(y_test_real[i], preds_test_real[i])
    sum_Dice += dice
avg_Dice = sum_Dice / 519
#The final average dice similarity coefficients on the test size
print(avg_Dice)

#Visualization for different size images
fig, axs = plt.subplots(3,3,figsize=(10,10))
fig.suptitle('Image Segmentation on different size images')
axs[0,0].imshow(X_test_real[2])
axs[0,1].imshow(y_test_real[2],cmap='gray')
axs[0,2].imshow(preds_test_real[2],cmap='gray')

axs[1,0].imshow(X_test_real[5])
axs[1,1].imshow(y_test_real[5],cmap='gray')
axs[1,2].imshow(preds_test_real[5],cmap='gray')

axs[2,0].imshow(X_test_real[12])
axs[2,1].imshow(y_test_real[12],cmap='gray')
axs[2,2].imshow(preds_test_real[12],cmap='gray')


axs[0,0].title.set_text('Original Image')
axs[0,1].title.set_text('Ground Truth')
axs[0,2].title.set_text('Predicion')

#Visualization for interesting findins
fig, axs = plt.subplots(3,figsize=(10,10),constrained_layout = True)
fig.suptitle('Wrong Ground Truth')

axs[0].imshow(X_test_real[0])
axs[1].imshow(y_test_real[0],cmap='gray')
axs[2].imshow(preds_test_real[0],cmap='gray')

axs[0].title.set_text('Original Image')
axs[1].title.set_text('Ground Truth')
axs[2].title.set_text('Predicion')