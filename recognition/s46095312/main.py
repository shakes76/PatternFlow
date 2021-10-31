import os
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import UNet
import math
import os


""" FUNCTION TO COMPUTE DICE SCORE 
    Inputs: two numpy arrays 
"""


def dice_loss_function(a, b):
    aIntB = np.logical_and(a == 1, b == 1)
    return 2 * aIntB.sum() / (a.sum() + b.sum())


# Define global variables
TRAIN_SIZE = 0.8
VALIDATE_SIZE = 0.1
TRAIN_BATCH_SIZE = 10
VALIDATE_BATCH_SIZE = 10

# Allow GPU more memory access
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Get file paths
input_data_path = "F:\\data\\ISIC2018_Task1-2_Training_Input"
label_data_path = "F:\\data\\ISIC2018_Task1_Training_GroundTruth"

input_data_paths = os.listdir(input_data_path)
gt_data_paths = os.listdir(label_data_path)

if len(input_data_paths) != len(gt_data_paths):
    print("Mismatch in data / labels")

nData = len(input_data_paths)

nTrain = math.floor(TRAIN_SIZE * nData)
nVal = math.floor(VALIDATE_SIZE * nData)
nTest = nData - nTrain - nVal

print(nData, " total samples")
print("Training on ", nTrain, " samples")
print("Validating on ", nVal, " samples")
print("Testing on ", nTest, " samples")


# make dimensions as divisible by powers of two as possible while maintaining the aspect ratio of the input
# (256 = 2^8, 192 = 3*2^6)
nx = 256
ny = 192

# instantiate and compile the network here
network = UNet(nx, ny)
# network.compileModel()

# binary segmentation
nTypes = 2

# first load all data and labels into a single array to shuffle before splitting into train/val/test sets
data = np.empty([nData, ny, nx, 3 + nTypes])

# allocate train,val,test data and label arrays here
trainData = np.empty([nTrain, ny, nx, 3])
valData = np.empty([nVal, ny, nx, 3])
testData = np.empty([nTest, ny, nx, 3])

trainLabels = np.empty([nTrain, ny, nx, nTypes])
valLabels = np.empty([nVal, ny, nx, nTypes])
testLabels = np.empty([nTest, ny, nx, nTypes])

# load all data and labels into the data array
for i, path in enumerate(input_data_paths):
    # report progress on loading images so we know it's not stalled
    if i % 200 == 0:
        print("loading input data: ", i)
    # open input image with Pillow
    input_img = Image.open(input_data_path + "\\" + path).resize((nx, ny))

    # normalise channels to 0-1 range. Place this data in cells 0-3 of the last axis of the total data array
    data[i, :, :, 0:3] = np.asarray(input_img) / 255.0

    gt_img = Image.open(label_data_path + "\\" + gt_data_paths[i]).resize((nx, ny))
    data[i, :, :, 3:5] = np.eye(nTypes)[(np.asarray(gt_img) * (1/255)).astype(int)]

# shuffle the data array
np.random.shuffle(data)

# split the data array into train, val, test input data and labels
trainData = data[0:nTrain, :, :, 0:3]
valData = data[nTrain:nTrain+nVal, :, :, 0:3]
testData = data[nTrain+nVal:, :, :, 0:3]
trainLabels = data[0:nTrain, :, :, 3:5]
valLabels = data[nTrain:nTrain+nVal, :, :, 3:5]
testLabels = data[nTrain+nVal:, :, :, 3:5]

# train the network
history = network.fit(trainData, trainLabels, valData, valLabels, 30)


# plot the performance characteristics during training. Accuracy (dice coefficient metric) and loss function at each
# epoch.
plt.plot(history['dice_metric'], label="Training Dice Coefficient")
plt.plot(history['val_dice_metric'], label="Validation Dice Coefficient")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.figure()
plt.plot(history['loss'], label="Training loss")
plt.plot(history['val_loss'], label="Validation loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend()

# compute test data prediction
testPredictions = network.predict(testData)

# compute the average dice score over the whole test set
avgDice = 0.0
for i in range(nTest):
    avgDice += dice_loss_function(testPredictions[i, :, :, :].argmax(axis=2), testLabels[i, :, :, :].argmax(axis=2))
avgDice /= nTest
print("Average dice score on test set is: ", avgDice)

# do a couple of specific sample predictions for visualisation purposes
samplePrediction_1 = network.predict(testData[0:1])
samplePrediction_2 = network.predict(testData[10:11])

# output the dice score of the example predictions
print("Dice score example 1: ", dice_loss_function(samplePrediction_1[0, :, :, :].argmax(axis=2), testLabels[0, :, :, :].argmax(axis=2)))
print("Dice score example 2: ", dice_loss_function(samplePrediction_2[0, :, :, :].argmax(axis=2), testLabels[10, :, :, :].argmax(axis=2)))

# plot the input images and output segmentation maps of the examples
plt.figure()
plt.imshow(testData[0])
plt.figure()
plt.imshow(samplePrediction_1[0,:,:,:].argmax(axis=2))

plt.figure()
plt.imshow(testData[10])
plt.figure()
plt.imshow(samplePrediction_2[0,:,:,:].argmax(axis=2))
plt.show()
