'''
Pattern Recognition
Segment the ISICs data set with the UNet

@author Yi Yang
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import glob
from tensorflow.keras.layers import concatenate, Flatten
from tensorflow.keras.layers import Input, Dense, Conv2D, UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model

# load files
filelist_input = glob.glob("C:/Users/s4520948/Downloads/ISIC2018_Task1-2_Training_Input_x2/*.jpg")
filelist_ground_truth = glob.glob("C:/Users/s4520948/Downloads/ISIC2018_Task1_Training_GroundTruth_x2/*.png")

# check size
print(len(filelist_input))
print(len(filelist_ground_truth))

# convert input image to array
x = convert_array(filelist_input)

# rescale
x = x / 255.

# convert ground truth to array
y = convert_array_truth(filelist_ground_truth)

# one hot
y = np.round(y / 255)

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# check shapes
print("x.shape:", x.shape)
print("y.shape:", y.shape)
print("x_train.shape:", x_train.shape)
print("x_test.shape:", x_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

# train model
model = model()

# fit
fit(model,x_train,y_train,10,4)

# predict
pred = model.predict(x_test,batch_size=4)
pred_mask = np.round(pred)

# calculate dice coefficient
dice = dice_coef(y_test, pred_mask, smooth=1.)

# display 9 results
for i in range(9):
    display([x_train[i],y_train[i],pred_mask[i]])
