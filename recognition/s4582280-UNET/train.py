import os
import numpy as np
import cv2
import tensorflow as tf
from glob import glob
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras import backend as K




"""
#train.py
# All the code for training and testing the model
from dataset import load_isic
from modules import BuildUNET
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 18
BATCH_SIZE = 64
TRAIN_LENGTH = 500
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the data from file
t, e, tr, et = load_isic(size=0.25)
print(t.shape, tr.shape)
tf.convert_to_tensor(t)
tf.convert_to_tensor(tr)

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")
    plt.show()
    
class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, smooth=1e-6, gama=2):
        super(DiceLoss, self).__init__()
        self.name = 'NDL'
        self.smooth = smooth
        self.gama = gama

    def call(self, y_true, y_pred):
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        return result

def DiceCoefLoss(y_true, y_pred):
    return -DiceLoss(y_true, y_pred)

#display([t[0], tr[0]])

model = BuildUNET()
model.compile(optimizer=tf.keras.optimizers.Adam(),loss=[DiceLoss],metrics="accuracy")
history = model.fit(t, tr, epochs=EPOCHS, steps_per_epoch=8)
"""
