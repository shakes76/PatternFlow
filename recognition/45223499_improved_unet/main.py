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
        img = cv2.imread(img)
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
