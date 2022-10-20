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
from modules import BuildResUnet
from dataset import FullLoad

# Parameters
smooth = 1e-14

# Dice coefficient
def DiceCoef(trueY, predY):
    trueY = tf.keras.layers.Flatten()(trueY)
    predY = tf.keras.layers.Flatten()(predY)
    intersection = tf.reduce_sum(predY * trueY)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def DiceLoss(trueY, predY):
    return 1 - DiceCoef(trueY, predY)

# Main function loop
if __name__ == "__main__":
    # Randomizing the seed for tensorflow and numpy
    tf.random.set_seed(42)
    np.random.seed(42)

    # Create a files directory if need be
    if not os.path.exists("files"):
        os.makedirs("files")

    # Parameters for the project
    lr = 1e-5
    epochs=4
    H, W = 256, 256
    modelPath = os.path.join("files", "model.h5")
    csvPath = os.path.join("files", "data.csv")

    # Loading the dataset
    path = "./Data/"
    trainDataset, testDataset, validDataset = FullLoad("./Data/")
    trainSteps = len(trainDataset)
    validSteps = len(validDataset)

    # Implementing the model
    model = BuildResUnet((H, W, 3))
    metrics = [DiceCoef, Recall(), Precision()]

    # Build the model
    model.compile(optimizer=Adam(lr), loss=DiceLoss, metrics=metrics)

    # Fit the model
    model.fit(
        trainDataset, epochs=epochs, validation_data=validDataset,
        steps_per_epoch=trainSteps, validation_steps=validSteps,
    )    

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
