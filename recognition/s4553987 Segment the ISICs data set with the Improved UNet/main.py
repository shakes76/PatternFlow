import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import tensorflow as tf
from random import shuffle, seed
from tensorflow.keras.layers import Conv2D, Activation, BatchNormalization, \
    Dropout, Input, concatenate, Add, UpSampling2D, Conv2DTranspose

inputs = [cv2.imread(file) for file in glob.glob('Downloads/ISIC2018_Task1-2_Training_Input_x2/*.jpg')]
outputs = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob('Downloads/ISIC2018_Task1_Training_GroundTruth_x2/*.png')]

for i in range(len(inputs)):
    inputs[i] = cv2.resize(inputs[i],(256,256))/255

for i in range(len(outputs)):
    outputs[i] = cv2.resize(outputs[i],(256,256))/255
    outputs[i][outputs[i] > 0.5] = 1
    outputs[i][outputs[i] <= 0.5] = 0

X = np.zeros([2594, 256, 256, 3])
y = np.zeros([2594, 256, 256])

for i in range(len(inputs)):
    X[i] = inputs[i]

for i in range(len(outputs)):
    y[i] = outputs[i]
        

y = y[:, :, :, np.newaxis]

X_train = X[0:1800,:,:,:]
X_val = X[1800:2197,:,:,:]
X_test = X[2197:2594,:,:,:]

y_train = y[0:1800,:,:,:]
y_val = y[1800:2197,:,:,:]
y_test = y[2197:2594,:,:,:]

model = improved_unet(256)

#### Compile Model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[dsc, 'accuracy'])


model_callback=tf.keras.callbacks.ModelCheckpoint(filepath='best_checkpoint',
                                              save_best_only= True,
                                              save_weights_only=True,
                                              monitor = 'val_accuracy',
                                              mode='max')

history = model.fit(x = X_train, y=y_train, epochs=30, verbose=1,
                    validation_data=(X_val, y_val), batch_size = 8, callbacks=[model_callback])

def dsc(y_true, y_pred):
    intersection = tf.reduce_sum(tf.math.multiply(y_true, y_pred))
    dsc = 2*intersection / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred))
    return dsc


