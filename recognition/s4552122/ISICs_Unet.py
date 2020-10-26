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
from tensorflow import keras
tf.random.Generator = None
import tensorflow_addons as tfa
import cv2
from sklearn.model_selection import train_test_split


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

