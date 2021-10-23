"""
Driver Script
"""

import tensorflow as tf
from model import *
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt

############LOAD CELEB-A####################
dir = "C:/ISIC_IMAGE_TEST"
mask_dir = "C:/ISIC_MASK_TEST"
batch_size = 16
img_height = 640
img_width = 480

y_train = tf.keras.utils.image_dataset_from_directory(mask_dir, validation_split=0.2,
  subset="training",
  labels=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode='grayscale')

y_val = tf.keras.utils.image_dataset_from_directory(mask_dir, validation_split=0.2,
  subset="validation",
  labels=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode='grayscale')

X_train = tf.keras.utils.image_dataset_from_directory(dir, validation_split=0.2,
  subset="training",
  labels=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode='grayscale')

X_val = tf.keras.utils.image_dataset_from_directory(dir, validation_split=0.2,
  subset="validation",
  labels=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size,
  color_mode='grayscale')


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images in y_train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
    plt.axis("off")

plt.show()

plt.figure(figsize=(10, 10))
for images in X_train.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"), cmap='gray')
    plt.axis("off")

plt.show()


#no_train = 15
#no_test         = 5
#images       = np.sort(os.listdir(diretory))
### name of the jpg files for training set
#train_images = images[:no_train]
### name of the jpg files for the testing data
#test_images  = images[no_train:no_train + no_test]
#image_res     = (640, 480, 3)
#
#masks = np.sort(os.listdir(mask_dir))
#
#train_masks = masks[:no_train]
### name of the jpg files for the testing data
#test_masks  = masks[no_train:no_train + no_test]
#
#def images_to_array(direc, nm_imgs_train):
#    images_array = []
#    for null, image_no in enumerate(nm_imgs_train):
#        image = load_img(direc + "/" + image_no,
#                         target_size=(64, 64))
#        image = img_to_array(image)/255
#        images_array.append(image)
#    images_array = np.array(images_array)
#    return images_array

#X_train = images_to_array(diretory, train_images)
#print("X_train.shape = {}".format(X_train.shape))
#
#X_test  = images_to_array(diretory, test_images)
#print("X_test.shape = {}".format(X_test.shape))
#
#y_train = images_to_array(mask_dir, train_masks)
#
#y_test = images_to_array(mask_dir, train_masks)
#
#fig = plt.figure(figsize=(30,10))
#nplot = 7
#for count in range(1,nplot):
#    ax = fig.add_subplot(1,nplot,count)
#    ax.imshow(X_train[count])
#plt.show()