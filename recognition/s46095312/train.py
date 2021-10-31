# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 18:15:43 2019

@author: Reza Azad
"""
from __future__ import division
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import model as M
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras import callbacks
import pickle
from model import IUNet
import time
import matplotlib.pyplot as plt
import tensorflow as tf


def dice_loss_function(a, b):
    aIntB = np.logical_and(a == 1, b == 1)
    return 2 * aIntB.sum() / (a.sum() + b.sum())


# ===== normalize over the dataset
def dataset_normalized(imgs):
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

####################################  Load Data #####################################
tr_data = np.load('data_train.npy')
te_data = np.load('data_test.npy')
val_data = np.load('data_val.npy')

tr_mask = np.load('mask_train.npy')
te_mask = np.load('mask_test.npy')
val_mask = np.load('mask_val.npy')

# tr_mask = np.expand_dims(tr_mask, axis=3)
# te_mask = np.expand_dims(te_mask, axis=3)
# val_mask = np.expand_dims(val_mask, axis=3)

print('ISIC18 Dataset loaded')

tr_data = dataset_normalized(tr_data)
te_data = dataset_normalized(te_data)
val_data = dataset_normalized(val_data)

print('dataset Normalized')

# Build and compile model
model = IUNet(256, 192)
model.my_compile()
# model = M.unet_model(input_size=(256, 256, 3))


print('Training')
batch_size = 10
nb_epoch = 25

mcp_save = ModelCheckpoint('weight_isic18', save_weights_only=True, save_best_only=True, monitor='dice_coef', mode='min')
# reduce_lr_loss = ReduceLROnPlateau(monitor='dice_coef', factor=0.85, patience=7, verbose=1, min_delta=1e-4, mode='min')
# fit model
tic = time.time()
model.my_fit(tr_data, tr_mask, val_data, val_mask,
             batch_size=batch_size, epochs=nb_epoch, callback=mcp_save)
history = model.history
print('Trained model saved', 'time:', time.time() - tic)

with open('hist_isic18', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Save the weights
model.my_save('./checkpoints/my_checkpoint')

# compute test data prediction
testPredictions = model.predict(te_data)

# do a couple of specific sample predictions for visualisation purposes
samplePrediction_1 = model.predict(te_data[0:1])
samplePrediction_2 = model.predict(te_data[10:11])

# Re-evaluate the model
loss, dice_coef, val_dice_coef = model.evaluate(te_data, te_mask)
print("Restored model, loss: {:5.2f}%".format(100 * loss))
print("Restored model, dice_coef: {:5.2f}%".format(100 * dice_coef))
print("Restored model, val_dice_coef: {:5.2f}%".format(100 * val_dice_coef))

# output the dice score of the example predictions
print("Dice score example 1: ",
      dice_loss_function(samplePrediction_1[0, :, :, :].argmax(axis=2), te_mask[0, :, :, :].argmax(axis=2)))
print("Dice score example 2: ",
      dice_loss_function(samplePrediction_2[0, :, :, :].argmax(axis=2), te_mask[10, :, :, :].argmax(axis=2)))
# plot the input images and output segmentation maps of the examples
plt.figure()
plt.imshow(te_data[0])
plt.figure()
plt.imshow(samplePrediction_1[0, :, :, :].argmax(axis=2))

plt.figure()
plt.imshow(te_data[10])
plt.figure()
plt.imshow(samplePrediction_2[0, :, :, :].argmax(axis=2))
plt.show()

# compute the average dice score over the whole test set
average_dice_cof = 0.0
for i in range(520-1):
    average_dice_cof += dice_loss_function(testPredictions[i, :, :, :].argmax(axis=2), te_mask[i, :, :, :].argmax(axis=2))
average_dice_cof /= 519
print("Average dice score on test set is: ", average_dice_cof)

