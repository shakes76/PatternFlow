# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 16:58:41 2022

@author: eudre
"""

import os
import numpy as np
import cv2
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
from tensorflow.python.client import device_lib



train_path ='C:/Users/eudre/test/ISIC-2017_Training_Data/*.jpg'
mask_path ='C:/Users/eudre/test/ISIC-2017_Training_Part1_GroundTruth/*.png'

def dice_coef(y_true, y_pred, epsilon=0.00001):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    
    """
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
batch_size = 4
lr = 1e-4 ## (0.0001)
num_epoch = 5


(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = dataset.load_data(train_path, mask_path)



train_dataset = dataset.tf_dataset(train_x, train_y, batch_size)
valid_dataset = dataset.tf_dataset(valid_x, valid_y, batch_size)

train_steps = len(train_x)//batch_size
valid_steps = len(valid_x)//batch_size

if len(train_x) % batch_size != 0:
    train_steps += 1

if len(valid_x) % batch_size != 0:
    valid_steps += 1

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
    TensorBoard(),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
]

# =============================================================================
# model = model.modified_UNET((256,256,3))
# model.compile(optimizer='adam', loss=dice_coef_loss, metrics=['accuracy', dice_coef])
# model_history = model.fit(train_dataset,
#                       epochs=num_epoch,
#                       validation_data=valid_dataset,
#                       steps_per_epoch=train_steps,
#                       validation_steps=valid_steps,
#                       callbacks=callbacks)
# =============================================================================
