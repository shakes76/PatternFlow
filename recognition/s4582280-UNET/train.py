#train.py
# All the code for training and testing the model
from dataset import load_isic
from modules import BuildUNET
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 18

# Load the data from file
t, e, tr, et = load_isic(size=0.05)
print(t.shape, tr.shape)


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")
    plt.show()

def DiceLoss(y_true, y_pred, smooth=1):
    # flatten
    y_true_f = keras.backend.flatten(y_true)
    y_pred_f = keras.backend.flatten(y_pred)
    # one-hot encoding y with 3 labels : 0=background, 1=label1, 2=label2
    y_true_f = keras.backend.one_hot(keras.backend.cast(y_true_f, np.uint8), 3)
    y_pred_f = keras.backend.one_hot(keras.backend.cast(y_pred_f, np.uint8), 3)
    # calculate intersection and union exluding background using y[:,1:]
    intersection = keras.backend.sum(y_true_f[:,1:]* y_pred_f[:,1:], axis=[-1])
    union = keras.backend.sum(y_true_f[:,1:], axis=[-1]) + keras.backend.sum(y_pred_f[:,1:], axis=[-1])
    # apply dice formula
    dice = keras.backend.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return 1 - dice

#display([t[0], tr[0]])

model = BuildUNET()
model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=[DiceLoss],
                  metrics="accuracy")
history = model.fit(t, tr, epochs=EPOCHS)
