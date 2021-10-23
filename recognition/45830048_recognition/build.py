"""
Driver Script
"""

import tensorflow as tf
from tensorflow.python.ops.gen_batch_ops import batch
from model import *
import numpy as np
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

############LOAD CELEB-A####################
dir = "C:/ISIC2018_Task1-2_Training_Input_x2"
mask_dir = "C:/ISIC2018_Task1_Training_GroundTruth_x2"
batchs = 16
img_height = 128
img_width = 128

y_train = tf.keras.utils.image_dataset_from_directory(mask_dir, validation_split=0.2,
  subset="training",
  labels=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batchs,
  color_mode='grayscale')

y_val = tf.keras.utils.image_dataset_from_directory(mask_dir, validation_split=0.2,
  subset="validation",
  labels=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batchs,
  color_mode='grayscale')

X_train = tf.keras.utils.image_dataset_from_directory(dir, validation_split=0.2,
  subset="training",
  labels=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batchs,
  color_mode='grayscale')

X_val = tf.keras.utils.image_dataset_from_directory(dir, validation_split=0.2,
  subset="validation",
  labels=None,
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batchs,
  color_mode='grayscale')

normalization_layer = tf.keras.layers.Rescaling(1./255)


X_train = X_train.unbatch()
X_val = X_val.unbatch()
y_train = y_train.unbatch()
y_val = y_val.unbatch()

test_size = int(0.2 * 2076)
X_test = X_train.take(test_size)
X_train = X_train.skip(test_size)
y_test = y_train.take(test_size)
y_train = y_train.skip(test_size)

a = tf.zeros([0, img_height, img_width, 1])
for image in X_train:
  image /= 255.0
  a = tf.concat([a, [image]], axis = 0)
X_train = a

a = tf.zeros([0, img_height, img_width, 1])
for image in X_val:
  image /= 255.0
  a = tf.concat([a, [image]], axis = 0)
X_val = a

a = tf.zeros([0, img_height, img_width, 1])
for image in X_test:
  image /= 255.0
  a = tf.concat([a, [image]], axis = 0)
X_test = a

a = tf.zeros([0, img_height, img_width, 1])
for image in y_train:
  image /= 255.0
  a = tf.concat([a, [image]], axis = 0)
y_train = a

a = tf.zeros([0, img_height, img_width, 1])
for image in y_val:
  image /= 255.0
  a = tf.concat([a, [image]], axis = 0)
y_val = a

a = tf.zeros([0, img_height, img_width, 1])
for image in y_test:
  image /= 255.0
  a = tf.concat([a, [image]], axis = 0)
y_test = a

print(y_train)

#plt.figure(figsize=(10, 10))
#for i in range(9):
#  ax = plt.subplot(3, 3, i + 1)
#  plt.imshow(y_train.numpy()[i].astype("uint8"), cmap='gray')
#  plt.axis("off")
#plt.show()
#
#
#plt.figure(figsize=(10, 10))
#for i in range(9):
#  ax = plt.subplot(3, 3, i + 1)
#  plt.imshow(X_train.numpy()[i].astype("uint8"), cmap='gray')
#  plt.axis("off")
#plt.show()

"""
Calculate dice coeffecient
"""
def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)

"""
dice coeffecient for use in loss function
"""
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

model = improved_unet(img_height, img_width)
model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = ['accuracy', dice_coef])
model.fit(X_train, y_train, validation_data =(X_val, y_val), epochs = 30, batch_size = 32)

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