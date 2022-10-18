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
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Load the data from file
t, e, tr, et = load_isic(size=0.05)
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

def DiceLoss(y_true, y_pred, smooth=100):        
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return dice

def DiceCoefLoss(y_true, y_pred):
    return 1-DiceLoss(y_true, y_pred)

#display([t[0], tr[0]])

model = BuildUNET()
model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=[DiceCoefLoss],
                  metrics="accuracy")
history = model.fit(t, tr, epochs=EPOCHS)
