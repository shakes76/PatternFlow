from __future__ import division
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle
from model import IUNet
import time
import matplotlib.pyplot as plt
import tensorflow as tf

smooth = 4e-6


def dice_loss_function(a, b):
    intersection = np.logical_and(a == 1, b == 1)
    return 2 * intersection.sum() / (a.sum() + b.sum() + smooth)


# normalize over the dataset
def dataset_normalized(images):
    std = np.std(images)
    mean = np.mean(images)
    normalization = (images - mean) / std
    for i in range(images.shape[0]):
        normalization[i] = ((normalization[i] - np.min(normalization[i])) / (
                np.max(normalization[i]) - np.min(normalization[i]))) * 255
    return normalization


# unlock memory limitation
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

print('ISIC18 Dataset loaded')

tr_data = dataset_normalized(tr_data)
te_data = dataset_normalized(te_data)
val_data = dataset_normalized(val_data)

print('dataset Normalized')

# Build and compile model
model = IUNet(256, 192)
model.my_compile()

print('Training')

mcp_save = ModelCheckpoint('weight_isic18',
                           save_weights_only=True,
                           save_best_only=True,
                           monitor='val_dice_coef',
                           mode='max')

# fit model
tic = time.time()
model.my_fit(tr_data, tr_mask, val_data, val_mask,
             batch_size=18,
             epochs=22,
             steps=None,
             callback=mcp_save)
history = model.history
print('Trained model saved', 'time:', time.time() - tic)

with open('hist_isic18', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# Save the weights
model.my_save('./checkpoints/my_checkpoint')
# plot the performance characteristics during training
plt.figure()
plt.plot(model.history.history['dice_coef'], label="Training Dice Coefficient")
plt.plot(model.history.history['val_dice_coef'], label="Validation Dice Coefficient")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig('output/dice_coef.png')
plt.close()
plt.figure()
plt.plot(model.history.history['loss'], label="Training loss")
plt.plot(model.history.history['val_loss'], label="Validation loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()
plt.show()
plt.savefig('output/loss.png')
