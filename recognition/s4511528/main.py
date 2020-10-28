"""
An improved uNet for ISICs dataset segmentation

@author foolish li jia min
"""

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt
from model import build_model


# Load images
images = [cv2.imread(file) for file in glob.glob('Downloads/ISIC2018_Task1-2_Training_Input_x2/*.jpg')]
masks = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob('Downloads/ISIC2018_Task1_Training_GroundTruth_x2/*.png')]

width = 256
height = 256
channels = 3

# Resize and reshape the image dataset
for i in range(len(images)):
    images[i] = cv2.resize(images[i],(height,width))
    images[i] = images[i]/255


for i in range(len(masks)):
    masks[i] = cv2.resize(masks[i],(height,width))
    masks[i] = masks[i]/255
    masks[i][masks[i] > 0.5] = 1
    masks[i][masks[i] <= 0.5] = 0

X = np.zeros([2594, height, width, channels])
y = np.zeros([2594, height, width])

for i in range(len(images)):
    X[i] = images[i]

for i in range(len(masks)):
    y[i] = masks[i]
        
y = y[:, :, :, np.newaxis]


# Split train:validation:test = 6:2:2 ------->  1556:519:519
X_train = X[0:1556,:,:,:]
X_val = X[1556:1556+519,:,:,:]
X_test = X[1556+519:2594,:,:,:]

y_train = y[0:1556,:,:,:]
y_val = y[1556:1556+519,:,:,:]
y_test = y[1556+519:2594,:,:,:]


# Define the F1-score
def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    y_pred = tf.cast(tf.math.greater(y_pred, 0.5),tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Train the improved Unet model
if __name__ == "__main__":
    model = build_model()

    metric = ['acc', dice_coef, Recall(), Precision()]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metric)
    history = model.fit(x = X_train, y = y_train, validation_data=(X_val, y_val), batch_size=16, epochs=30)


# Evaluate model
results = model.evaluate(X_test, y_test, batch_size = 16)

# Prediction
predictions = model.predict(X_test)

# Show results
for i in range(len(predictions)):
    predictions[i][predictions[i] > 0.5] = 1
    predictions[i][predictions[i] <= 0.5] = 0

# Plot Origional/ Segmentation/ Prediction pictures

n = 10 
plt.figure(figsize=(30, 10))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(X_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.figure(figsize=(30, 10))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(y_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.figure(figsize=(30, 10))
for i in range(1, n+1):
    ax = plt.subplot(1, n, i)
    plt.imshow(predictions[i])
    plt.gray() 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

# Calculate average test dice coefficient
test_dice = dice_coef(y_test, predictions)
print('Test Dice Coefficient: ', test_dice.numpy())
