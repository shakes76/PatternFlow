import tensorflow as tf
import tf.keras
import os
import numpy as np
from PIL import Image
import torch

X_train_path = ".../Data_Files/ISIC-2017_Training_Data"
Y_train_path = ".../Data_Files/ISIC-2017_Training_Part1_GroundTruth"

print("Program starting")

X_train = []
for f in os.listdir(X_train_path):
    if f.endswith(".csv") or "superpixels" in f:
        continue
    x_train = Image.open(os.path.join(X_train_path, f)).resize((256, 256))
    x_train = np.array(x_train)
    x_train = x_train / 255.0
    # print(x_train.shape)
    X_train.append(x_train)

np.stack(X_train)
print("********* Data loading for X_training complete *********")

Y_train = []
for f in os.listdir(Y_train_path):
    if f.endswith(".csv") or "superpixels" in f:
        continue
    y_train = Image.open(os.path.join(Y_train_path, f)).resize((256, 256))
    y_train = np.array(y_train)
    # print(y_train.shape)
    Y_train.append(y_train)

np.stack(Y_train)

print(X_train.shape)
# print(X_train)
print(Y_train.shape)
