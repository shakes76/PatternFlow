#train.py
# All the code for training and testing the model
from dataset import load_isic

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

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
    
display([t[0], tr[0]])
