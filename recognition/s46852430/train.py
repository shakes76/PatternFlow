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
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import SGD
import dataset
import model
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt



train_path ='C:/Users/eudre/test/ISIC-2017_Training_Data/*.jpg'
mask_path ='C:/Users/eudre/test/ISIC-2017_Training_Part1_GroundTruth/*.png'

def dice_coef(y_true, y_pred, epsilon=0.00001):

    axis = (0,1,2,3)
    dice_numerator = 2. * backend.sum(y_true * y_pred, axis=axis) + epsilon
    dice_denominator = backend.sum(y_true*y_true, axis=axis) + backend.sum(y_pred*y_pred, axis=axis) + epsilon
    return backend.mean((dice_numerator)/(dice_denominator))

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)




""" Seeding """
np.random.seed(42)
tf.random.set_seed(42)

""" Hyperparameters """
batch_size = 32

num_epoch = 5

learning_rate = 0.1
decay_rate = learning_rate / num_epoch
momentum = 0.9

(train_set), (valid_set), (test_set) = dataset.spilt_data(train_path, mask_path)


training_set = train_set.map(dataset.load_data)
validation_set=valid_set.map(dataset.load_data)
test_set=test_set.map(dataset.load_data)

opt = SGD(learning_rate = learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)


callback = [
    ReduceLROnPlateau(monitor='loss', factor=0.1, patience=1, min_lr=1e-7, verbose=2),
    EarlyStopping(monitor='loss', patience = 1, restore_best_weights=False)
]


model = model.modified_UNET((256,256,3))
model.compile(optimizer= opt, loss=dice_coef_loss, metrics=[dice_coef])
#Train the model
model_history=model.fit(training_set.batch(batch_size), validation_data=validation_set.batch(batch_size), epochs=num_epoch, callbacks = callback)
#Evaluate
result=model.evaluate(test_set.batch(batch_size), verbose=1)

plt.figure(1, figsize=(20,10))
plt.plot(range(len(model_history.history['loss'])),model_history.history['loss'], label='loss')
plt.plot(range(len(model_history.history['dice_coef'])),model_history.history['dice_coef'], label='dice_coef')

plt.plot(range(len(model_history.history['val_loss'])),model_history.history['val_loss'], label='val_loss')
plt.plot(range(len(model_history.history['val_dice_coef'])),model_history.history['val_dice_coef'], label='val_dice_coef')
plt.legend()
plt.show()

model.save("Unet")


# =============================================================================
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True, rankdir='LR')
# =============================================================================
