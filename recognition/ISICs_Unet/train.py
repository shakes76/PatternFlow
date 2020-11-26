"""
File name: train.py
Author: Thomas Chen
Date created: 11/3/2020
Date last modified: 11/24/2020
Python Version: 3
"""
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras.callbacks import *

from data import train_generator, val_generator
from setting import *
from unet import UNet

"""
train data with unet provided with unet.py, data preprocessed with data.py and global setting set in the setting.py
"""
model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss', verbose=1, save_best_only=True)
model = UNet((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
hist = model.fit(
    train_generator,
    batch_size=BATCH_SIZE,
    steps_per_epoch=400,
    epochs=40,
    callbacks=[model_checkpoint],
    validation_data=val_generator,
    validation_steps=400,
    validation_batch_size=BATCH_SIZE)


def plot_history(history, key):
    train_hist = history[key]
    valid_hist = history['val_' + key]
    plt.figure()
    plt.plot(train_hist, 'r', label='train_' + key)
    plt.plot(valid_hist, 'b', label='valid_' + key)
    plt.title('train and valid ' + key)
    plt.xlabel('epochs')
    plt.ylabel(key)
    plt.legend()
    plt.savefig('train and valid ' + key + '.png')
    plt.show()


"""
plot the train history
"""
plot_history(hist.history, 'loss')
plot_history(hist.history, 'accuracy')
plot_history(hist.history, 'dice_coef')

shutil.copy("unet.hdf5", "unet_bak.hdf5")