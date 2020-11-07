## Include libraries

import numpy as np
import matplotlib.pyplot as plt

# Build network
from tensorflow.keras.layers import Dense,Reshape,Dropout,LeakyReLU,Flatten,BatchNormalization,Conv2D,Conv2DTranspose,MaxPool2D,MaxPooling2D,AveragePooling2D
from tensorflow.keras.models import Sequential
from neuralnet import *

# For image reading
import glob
from PIL import Image

# Tensorflow support
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)


## Reade images from folder
fileList = glob.glob("AKOA_Analysis/*.png")
X_trainr = np.array([np.array(Image.open(fname)) for fname in fileList])


## Image preporcessing
X_trainr = tf.image.resize(X_trainr,[128, 128]).numpy()
X_train = np.zeros((X_trainr.shape[0],128,128))
for x in range(X_trainr.shape[0]):
    X_train[x,:,:] =  (rgb2gray(X_trainr[x,:,:,:])/255.0-0.5)*2




## Variables
noise_size = 50
ksize = 3
poolsize = (2,2)
dropoutrate = 0.25
strid = 2
LeakyReluvalue = 0.2
learningr = 0.0003
beta_1v = 0.75
epochs = 100
batchsize = 32


## Design Generator
Generator = buildGen(noise_size,ksize, strid)

## Design Generator
Discriminator = buildDis(ksize, strid, poolsize)


## Build network
GAN = buildGAN(Generator,Discriminator, learningr, beta_1v) 


## Training function
GAN = trainnetwork(GAN, epochs, batchsize, X_train, noise_size)




