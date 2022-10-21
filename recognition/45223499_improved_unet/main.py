import tensorflow as tf
from matplotlib import image
import imageio
import os
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import PIL
import PIL.Image
import random
import cv2
import glob
from model.py import *

# download images
img_height =256
img_width = 256
imag_channels = 3
imag_input = "C:/Users/s4522349/Downloads/ISIC2018_Task1-2_Training_Input_x2/"
output = "C:/Users/s4522349/Downloads/ISIC2018_Task1_Training_GroundTruth_x2/"
imag_input = pathlib.Path(imag_input)
imag_output = pathlib.Path(output)

# list files
def lis_files(path, names):   
    lis = []
    for name in names:
        image = os.path.join(path, name)
        image = image.replace('\\', '/')
        lis += [image]
    return lis
image_name = os.listdir(imag_input)[1:2595]
list_input = lis_files(imag_input, image_name)
imag_output = os.listdir(imag_output)[1:2595]
list_output = lis_files(output, imag_output)

# divide dataset into train, test and validation datasets
train_X = list_input[:1558]
val_X = list_input[1558:2076]
test_X = list_input[2076:2594]
train_y = list_output[:1558]
val_y = list_output[1558:2076]
test_y = list_output[2076:2594]

# decode mask images and resize and round them
def decode_mask(masks):
    l = []
    for img in masks:
        img = image.imread(img)
        img = (img != 0).astype(np.uint8)
        img = cv2.resize(img, (img_height, img_width))
        l.append(img)
    return l

# decode images and resize and normalize them
def decode_img(imges):
    l = []
    for img in imges:
        img = image.imread(img)
        img = cv2.resize(img, (img_height, img_width))/255.0
        l.append(img)
    return l

train_X = np.asarray(decode_img(train_X))
val_X = np.asarray(decode_img(val_X))
test_X = np.asarray(decode_img(test_X))
train_y = np.asarray(decode_mask(train_y))
val_y = np.asarray(decode_mask(val_y))
test_y = np.asarray(decode_mask(test_y))
train_y = train_y[:, :, :, np.newaxis]
val_y = val_y[:, :, :, np.newaxis]
test_y = test_y[:, :, :, np.newaxis]

# plot train and loss in the model
def plot_model_history(model_history):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    plt.show()

# fit model
model = model(img_height, img_width, imag_channels)
history =  model.fit(train_X, train_y, validation_data =(val_X, val_y), batch_size = 16, epochs=5)
pred_test = model.predict(test_X)

#dice coefficient
def dice_coefficient(y_true, y_pred, smooth = 0):
    y_true = tf.cast(y_true, tf.float32)
    #change the dimension to one
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    #calculation for the loss function
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

# calcculate dice coefficient 
lis = 0
for i in range(518):  
    dice_coefficient_value = dice_coefficient(test_y[i], pred_test[i], smooth = 0)
    lis += dice_coefficient_value
ave = lis/518
print(ave)

# plot test images, masks and predict masks
#test_X
fig, ax = plt.subplots(3,3,figsize=(10, 10))
ax[0,0].imshow(test_X[12])
ax[0,1].imshow(test_y[12], cmap='gray')
ax[0,2].imshow(np.round(pred_test[12]), cmap='gray')
ax[1,0].imshow(test_X[13])
ax[1,1].imshow(test_y[13], cmap='gray')
ax[1,2].imshow(np.round(pred_test[13]), cmap='gray')
ax[2,0].imshow(test_X[0])
ax[2,1].imshow(test_y[0], cmap='gray')
ax[2,2].imshow(np.round(pred_test[0]), cmap='gray')
ax[0,0].title.set_text("images")
ax[0,1].title.set_text("ground truth")
ax[0,2].title.set_text("mask") 

