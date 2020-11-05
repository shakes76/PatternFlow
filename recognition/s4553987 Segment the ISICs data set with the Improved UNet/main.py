import cv2
import glob
import numpy as np

inputs = [cv2.imread(file) for file in glob.glob('Downloads/ISIC2018_Task1-2_Training_Input_x2/*.jpg')]
outputs = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob('Downloads/ISIC2018_Task1_Training_GroundTruth_x2/*.png')]

for i in range(len(inputs)):
    inputs[i] = cv2.resize(inputs[i],(256,256))/255

for i in range(len(outputs)):
    outputs[i] = cv2.resize(outputs[i],(256,256))/255
    outputs[i][outputs[i] > 0.5] = 1
    outputs[i][outputs[i] <= 0.5] = 0

X = np.zeros([2594, 256, 256, 3])
y = np.zeros([2594, 256, 256])

for i in range(len(inputs)):
    X[i] = inputs[i]

for i in range(len(outputs)):
    y[i] = outputs[i]
        

y = y[:, :, :, np.newaxis]

X_train = X[0:1800,:,:,:]
X_val = X[1800:2197,:,:,:]
X_test = X[2197:2594,:,:,:]

y_train = y[0:1800,:,:,:]
y_val = y[1800:2197,:,:,:]
y_test = y[2197:2594,:,:,:]