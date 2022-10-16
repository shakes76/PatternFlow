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

def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(display_list[i])
        plt.axis("off")
    plt.show()

def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = tf.keras.backend.flatten(inputs)
    targets = tf.keras.backend.flatten(targets)
    
    intersection = tf.keras.backend.sum(tf.keras.backend.dot(targets, inputs))
    dice = (2*intersection + smooth) / (tf.keras.backend.sum(targets) + tf.keras.backend.sum(inputs) + smooth)
    return 1 - dice

display([t[0], tr[0]])

model = BuildUNET()
model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=[DiceLoss],
                  metrics="accuracy")

history = model.fit(t, tr, epochs=EPOCHS)
