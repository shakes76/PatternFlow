# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:58:41 2022

@author: eudre
"""

import os
import numpy as np
from glob import glob
from IPython.display import clear_output
from tensorflow.keras import backend
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
import dataset
import model



train_path ='C:/Users/eudre/test/ISIC-2017_Training_Data/*.jpg'
mask_path ='C:/Users/eudre/test/ISIC-2017_Training_Part1_GroundTruth/*.png'

def dice_coef(y_true, y_pred, epsilon=0.00001):

    axis = (0,1,2,3)
    dice_numerator = 2. * backend.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = backend.sum(y_true*y_true, axis=axis) + backend.sum(y_pred*y_pred, axis=axis) + epsilon
    return backend.mean((dice_numerator)/(dice_denominator))

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)




""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

tf.config.list_physical_devices('GPU')
""" Hyperparameters """
batch_size = 32

num_epoch = 50


(train_set), (valid_set), (test_set) = dataset.spilt_data(train_path, mask_path)

training_set = train_set.map(dataset.load_data)
validation_set=valid_set.map(dataset.load_data)
test_set=test_set.map(dataset.load_data)




callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
]

model = model.modified_UNET((256,256,3))
model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy', dice_coef])
#Train the model
model_history=model.fit(training_set.batch(batch_size), validation_data=validation_set.batch(batch_size), epochs=60, callbacks = callbacks)
#Evaluate
result=model.evaluate(test_set.batch(batch_size), verbose=1)