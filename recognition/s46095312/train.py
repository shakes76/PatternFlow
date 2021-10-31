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
from model import UNet
import time
import matplotlib.pyplot as plt
import tensorflow as tf


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

tr_mask = np.expand_dims(tr_mask, axis=3)
te_mask = np.expand_dims(te_mask, axis=3)
val_mask = np.expand_dims(val_mask, axis=3)

print('ISIC18 Dataset loaded')

tr_data = dataset_normalized(tr_data)
te_data = dataset_normalized(te_data)
val_data = dataset_normalized(val_data)

tr_mask = tr_mask / 255.
te_mask = te_mask / 255.
val_mask = val_mask / 255.

print('dataset Normalized')

# Build and compile model
model = UNet(256, 256)
# model = M.unet_model(input_size=(256, 256, 3))


print('Training')
batch_size = 10
nb_epoch = 30

# mcp_save = ModelCheckpoint('weight_isic18', save_best_only=True, monitor='val_loss', mode='min')
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_delta=1e-4, mode='min')
# fit model
tic = time.time()
history = model.fit(tr_data, tr_mask, val_data, val_mask, epochs=30)
# history = model.fit(tr_data, tr_mask,
#                     batch_size=batch_size,
#                     epochs=nb_epoch,
#                     shuffle=True,
#                     verbose=1,
#                     validation_data=(val_data, val_mask), callbacks=[mcp_save, reduce_lr_loss])

print('Trained model saved', 'time:', time.time() - tic)

with open('hist_isic18', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Save the weights
model.save_weights('./checkpoints/my_checkpoint')
samplePrediction_1 = model.predict(te_data[0:1])

# plot the input images and output segmentation maps of the examples
plt.figure()
plt.imshow(te_data[0])
plt.figure()
plt.imshow(samplePrediction_1[0, :, :, :].argmax(axis=2))

plt.figure()
plt.imshow(te_data[10])
plt.figure()
plt.imshow(samplePrediction_1[0,:,:,:].argmax(axis=2))
plt.show()

