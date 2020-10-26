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
#The activation function is "leaky ReLe" which the alpha is 1e-2.
leakyRELU =tf.keras.layers.LeakyReLU(alpha=1e-2)
def context_modules(previous_layer, numFilters):
    """
    The context module:
    A pre-activation residual block with two 3x3 convolutional layers and a dropout layer (pdrop = 0.3) in between.
    @param previous_layer, the layer before the context module
    @numFilters, the number of output filters for every convolutional layer
    @return, the layer after the context module
    """
    
    l1 = tfa.layers.InstanceNormalization()(previous_layer)
    l2 = tf.keras.layers.Activation("relu")(l1)
    l3 = tf.keras.layers.Conv2D(numFilters, (3, 3), activation = leakyRELU, padding="same")(l2)
    l4 = tfa.layers.InstanceNormalization()(l3)
    l5 = tf.keras.layers.Activation("relu")(l4)
    l6 = tf.keras.layers.Conv2D(numFilters, (3, 3), activation = leakyRELU, padding="same")(l5)
    l6 = tf.keras.layers.Dropout(0.3)(l6)
    
    return l6

def upSampling(previous_layer, numFilters):
    """
    Upsampling the low resolution feature maps:
    A simple upscale that repeats the feature voxels twice in each spatial dimension, 
    followed by a 3x3 convolution that halves the number of feature maps.
    @param previous_layer, the layer before the Upsampling
    @numFilters, the number of output filters for the convolutional layer
    @return, the layer after the upSampling module
    """
    
    l1 = tf.keras.layers.UpSampling2D()(previous_layer)
    l2 = tf.keras.layers.Conv2D(numFilters,(3,3), activation = leakyRELU, padding="same")(l1)
    
    return l2

def localization(previous_layer, numFilters):
    """
    A localization module consists of a 3x3x3 convolution followed by a 1x1 convolution 
    that halves the number of feature maps.
    @param previous_layer, the layer before the localization module
    @numFilters, the number of output filters for every convolutional layer
    @return, the layer after the localization module
    """
    
    l1 = tf.keras.layers.Conv2D(numFilters, (3, 3), activation = leakyRELU, padding="same")(previous_layer)
    l2 = tf.keras.layers.Conv2D(numFilters, (1, 1), activation = leakyRELU, padding="same")(l1)
    
    return l2

#Model
inputs = tf.keras.layers.Input((256, 256, 3))

#Encode path
c1 = tf.keras.layers.Conv2D(16, (3, 3), activation = leakyRELU, padding="same")(inputs)
c2 = context_modules(c1, 16)
c3 = tf.keras.layers.Add()([c1, c2])

c4 = tf.keras.layers.Conv2D(32, (3, 3),activation = leakyRELU, strides = 2, padding="same")(c3)
c5 = context_modules(c4, 32)
c6 = tf.keras.layers.Add()([c4, c5])

c7 = tf.keras.layers.Conv2D(64, (3, 3),activation = leakyRELU, strides = 2, padding="same")(c6)
c8 = context_modules(c7, 64)
c9 = tf.keras.layers.Add()([c7, c8])

c10 = tf.keras.layers.Conv2D(128, (3, 3),activation = leakyRELU, strides = 2, padding="same")(c9)
c11 = context_modules(c10, 128)
c12 = tf.keras.layers.Add()([c10, c11])


c13 = tf.keras.layers.Conv2D(256, (3, 3),activation = leakyRELU, strides = 2, padding="same")(c12)
c14 = context_modules(c13, 256)
c15 = tf.keras.layers.Add()([c13, c14])

#Decode path
c16 = upSampling(c15, 128)
c17 = tf.keras.layers.concatenate([c16, c12])

c18 = localization(c17, 128)
c19 = upSampling(c18, 64)
c20 = tf.keras.layers.concatenate([c19, c9])

c21 = localization(c20, 64)
s1 = tf.keras.layers.Conv2D(2, (1, 1), activation = leakyRELU, padding="same")(c21)
s1 = tf.keras.layers.UpSampling2D(interpolation = "bilinear")(s1)

c23 = upSampling(c21, 32)
c24 = tf.keras.layers.concatenate([c23, c6])
c25 = localization(c24, 32)
s2 = s1 = tf.keras.layers.Conv2D(2, (1, 1), activation = leakyRELU, padding="same")(c25)
s3 = tf.keras.layers.Add()([s1, s2])
s3 = tf.keras.layers.UpSampling2D(interpolation = "bilinear")(s2)

c27 = upSampling(c25, 16)
c28 = tf.keras.layers.concatenate([c27, c3])

c29 = tf.keras.layers.Conv2D(32, (3, 3), activation = leakyRELU, padding='same')(c28)
s4 = tf.keras.layers.Conv2D(2, (1, 1), activation = leakyRELU, padding="same")(c29) 
s5 = tf.keras.layers.Add()([s3, s4])

outputs = tf.keras.activations.sigmoid(s5)
model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.summary()
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
results = model.fit(X_train, y_train, validation_data= (X_val, y_val), batch_size=16, epochs=5)