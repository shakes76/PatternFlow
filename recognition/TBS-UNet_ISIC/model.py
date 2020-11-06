import tensorflow as tf
from tensorflow.keras import datasets, layers, models

""" This part goes in the driver script """
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

# Get filepaths
input_data_dir = "data\\ISIC2018_Task1-2_Training_Input_x2"
gt_data_dir = "data\\ISIC2018_Task1_Training_GroundTruth_x2"

input_data_paths = os.listdir(input_data_dir)
gt_data_paths = os.listdir(gt_data_dir)

if(len(input_data_paths) != len(gt_data_paths)):
    print("Mismatch in data / labels")

nData = len(input_data_paths)
trainProp = 0.8
valProp = 0.1
testProp = 0.1
nTrain = math.floor(trainProp * nData)
nVal = math.floor(valProp * nData)
nTest = nData - nTrain - nVal

print(nData, " total samples")
print("Training on ", nTrain, " samples")
print("Validating on ", nVal, " samples")
print("Testing on ", nTest, " samples")


# make dimensions as divisible by powers of two as possible while maintaining the aspect ratio of the input
# (512 = 2^9, 384 = 3*2^7)
nx = 256
ny = 192
nTypes = 2

# first load all data and labels into a single array
data = np.empty([nData, ny, nx, 3 + nTypes])

trainData = np.empty([nTrain, ny, nx, 3])
valData = np.empty([nVal, ny, nx, 3])
testData = np.empty([nTest, ny, nx, 3])

trainLabels = np.empty([nTrain, ny, nx, nTypes])
valLabels = np.empty([nVal, ny, nx, nTypes])
testLabels = np.empty([nTest, ny, nx, nTypes])

# load all data and labels into the data array
for i, path in enumerate(input_data_paths):
    if i % 200 == 0:
        print("loading input ", i)
    input_img = Image.open(input_data_dir + "\\" + path).resize((nx, ny))
    data[i,:,:,0:3] = np.asarray(input_img) / 255.0

    gt_img = Image.open(gt_data_dir + "\\" + gt_data_paths[i]).resize((nx, ny))
    data[i,:,:,3:5] = np.eye(nTypes)[(np.asarray(gt_img) * (1/255)).astype(int)]

# shuffle the data array
np.random.shuffle(data)
trainData = data[0:nTrain, :, :, 0:3]
valData = data[nTrain:nTrain+nVal, :, :, 0:3]
testData = data[nTrain+nVal:, :, :, 0:3]
trainLabels = data[0:nTrain, :, :, 3:5]
valLabels = data[nTrain:nTrain+nVal, :, :, 3:5]
testLabels = data[nTrain+nVal:, :, :, 3:5]



"""
# iterate over data to construct full input data
for i, path in enumerate(input_data_paths):
    if i % 200 == 0:
        print("loading input ", i)
    img = Image.open(input_data_dir + "\\" + path).resize((nx, ny))
    # split data into the 3 data sets - done this way so we have no excess mem allocation
    if i < nTrain:
        trainData[i] = np.asarray(img) / 255.0
    elif i >= nTrain and i < nTrain + nVal:
        valData[i - nTrain] = np.asarray(img) / 255.0
    else:
        testData[i - nTrain - nVal] = np.asarray(img) / 255.0


# iterate over data to construct full input labels
for i, path in enumerate(gt_data_paths):
    if i % 200 == 0:
        print("loading labels ", i)
    img = Image.open(gt_data_dir + "\\" + path).resize((nx, ny))
    if i < nTrain:
        trainLabels[i] = np.eye(nTypes)[(np.asarray(img) * (1/255)).astype(int)]
    elif i >= nTrain and i < nTrain + nVal:
        valLabels[i - nTrain] = np.eye(nTypes)[(np.asarray(img) * (1/255)).astype(int)]
    else:
        testLabels[i - nTrain - nVal] = np.eye(nTypes)[(np.asarray(img) * (1/255)).astype(int)]
"""

""" FUNCTION TO COMPUTE DICE SCORE 
    Inputs: two numpy arrays 
"""
def diceScore(a, b):
    aIntB = np.logical_and(a == 1, b == 1)
    return 2 * aIntB.sum() / (a.sum() + b.sum())

# TEST - please delete
print("test dice score, should be 1: ", diceScore(trainLabels[100,:,:,:].argmax(axis=2), trainLabels[100,:,:,:].argmax(axis=2)))
print("test dice score, should be < 1: ", diceScore(trainLabels[100,:,:,:].argmax(axis=2), trainLabels[200,:,:,:].argmax(axis=2)))

""" This part goes in the model script"""

### STANDARD U-NET
# down path

# down-level 1
input_layer = layers.Input((ny,nx,3))
conv_1 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(input_layer)
conv_2 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(conv_1)
mPool_1 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_2)

# down-level 2
conv_3 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(mPool_1)
conv_4 = layers.Conv2D(128, (3,3), activation='relu', padding='same')(conv_3)
mPool_2 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_4)

# down-level 3
conv_5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(mPool_2)
conv_6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_5)
mPool_3 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_6)

# down-level 4
conv_7 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(mPool_3)
conv_8 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_7)
mPool_4 = layers.MaxPooling2D(pool_size=(2,2), padding='same')(conv_8)

# bottom-level
conv_9 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(mPool_4)
conv_10 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv_9)

# up path
# up-level 4
uconv_1 = layers.Conv2DTranspose(512, (2,2), strides=(2,2), padding='same')(conv_10)
cat_1 = layers.Concatenate()([uconv_1,conv_8])
conv_11 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(cat_1)
conv_12 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv_11)

# up-level 3
uconv_2 = layers.Conv2DTranspose(256, (2,2), strides=(2,2), padding='same')(conv_12)
cat_2 = layers.Concatenate()([uconv_2, conv_6])
conv_13 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(cat_2)
conv_14 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv_13)

# up-level 2
uconv_3 = layers.Conv2DTranspose(128, (2, 2), strides=(2,2), padding='same')(conv_14)
cat_3 = layers.Concatenate()([uconv_3, conv_4])
conv_15 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(cat_3)
conv_16 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv_15)

# up-level 1
uconv_4 = layers.Conv2DTranspose(64, (2, 2), strides=(2,2), padding='same')(conv_16)
cat_4 = layers.Concatenate()([uconv_4, conv_2])
conv_17 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(cat_4)
conv_18 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv_17)
conv_19 = layers.Conv2D(2, (1, 1), activation='softmax', padding='same')(conv_18)

unet = tf.keras.Model(inputs=input_layer, outputs=conv_19)
unet.compile(optimizer='adam', loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])
print(unet.summary())

h = unet.fit(trainData, trainLabels, validation_data=(valData, valLabels), batch_size=16, epochs=3)




""" PLOTTING STUFF - Should be in driver script i think"""
plt.plot(h.history['accuracy'])
plt.plot(h.history['val_accuracy'])

samplePrediction_1 = unet.predict(trainData[100:101])
samplePrediction_2 = unet.predict(trainData[200:201])

# compute test data prediction
testPredictions = unet.predict(testData)

#
avgDice = 0.0
for i in range(nTest):
    avgDice += diceScore(testPredictions[i,:,:,:].argmax(axis=2), testLabels[i,:,:,:].argmax(axis=2))
avgDice /= nTest
print("Average dice score on test set is: ", avgDice)
#print("Dice score example 1: ", diceScore(samplePrediction_1[0,:,:,:].argmax(axis=2), trainLabels[100,:,:,:].argmax(axis=2)))
#print("Dice score example 2: ", diceScore(samplePrediction_2[0,:,:,:].argmax(axis=2), trainLabels[200,:,:,:].argmax(axis=2)))

plt.figure()
plt.imshow(trainData[100])
plt.figure()
plt.imshow(samplePrediction_1[0,:,:,0])
plt.figure()
plt.imshow(samplePrediction_1[0,:,:,1])
plt.figure()
plt.imshow(samplePrediction_1[0,:,:,:].argmax(axis=2))

plt.figure()
plt.imshow(trainData[200])
plt.figure()
plt.imshow(samplePrediction_2[0,:,:,0])
plt.figure()
plt.imshow(samplePrediction_2[0,:,:,1])
plt.figure()
plt.imshow(samplePrediction_2[0,:,:,:].argmax(axis=2))
plt.show()
