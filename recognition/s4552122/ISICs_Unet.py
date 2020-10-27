"""
ISICs dataset by an improved Unet to make image segment.

@author Xiaoqi Zhuang
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
from model.py import improvedUnet


#loading features/original images
X = []
Path = "ISIC2018_Task1-2_Training_Input_x2"
for img in os.listdir(Path):
    im = image.imread("ISIC2018_Task1-2_Training_Input_x2/" + img) 
    X.append(im)
X = np.array(X)
plt.imshow(X[0])
#Since images have different shapes, I resize them to 256*256 images.
for i in range(len(X)):
    X[i] = cv2.resize(X[i],(256,256))
plt.imshow(cv2.resize(X[0],(256,256)))
#Reshape the image to be (number of images, 256, 256, 3)
images = np.zeros([2594,256,256,3])
for i in range(len(X)):
    images[i] = X[i]
images = images/255

#Loading segmentation images
Y = []
Path = "ISIC2018_Task1_Training_GroundTruth_x2"
for img in os.listdir(Path):
    im = image.imread("ISIC2018_Task1_Training_GroundTruth_x2/" + img) 
    Y.append(im)
for i in range(len(Y)):
    Y[i] = cv2.resize(Y[i],(256,256))
Y = np.array(Y)[:, :, :, np.newaxis]
#Make the segmentation images' labels just be 0 and 1.
Y = np.around(Y)
plt.imshow(images[1])
plt.imshow(Y[1],cmap="gray")

#Split training set, validation set, testing set.
X_tv, X_test, y_tv, y_test = train_test_split(images, Y, test_size = 0.2, random_state = 33)
X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size = 0.2, random_state = 33)
print("X_train shape: ", X_train.shape)
print("X_val shape: ", X_val.shape)
print("X_test shape: ", X_test.shape)
print("y_train shape: ", y_train.shape)
print("y_val shape: ", y_val.shape)
print("y_test shape: ", y_test.shape)

#Building the improved uNet model
model = improvedUnet()
model.summary()
results = model.fit(X_train, y_train, validation_data= (X_val, y_val), batch_size=16, epochs=5)

#Prediction
preds_test = model.predict(X_test, verbose=1)
preds_test = tf.math.argmax(preds_test, -1)
fig, axs = plt.subplots(2,2,figsize=(10,10))
#fig.suptitle('Predicion, Ground Truth, Original Image')
axs[0,0].imshow(X_test[2])
axs[0,1].imshow(y_test[2],cmap='gray')
axs[1,0].imshow(preds_test[2],cmap='gray')
axs[-1, -1].axis('off')

axs[0,0].title.set_text('Original Image')
axs[0,1].title.set_text('Ground Truth')
axs[1,0].title.set_text('Predicion')

#Evaluation: Dice similarity coefficient on the test set
y_test = y_test.reshape([519,256,256])
def dice_similar_coef(y_true, y_pred):
    
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)
sum_Dice = 0
avg_Dice = 0
for i in range(y_test.shape[0]):
    dice = dice_similar_coef(y_test[i], preds_test[i])
    sum_Dice += dice
avg_Dice = sum_Dice / 519
#avg_Dice is the average Dice similarity coefficient on the test set.
avg_Dice