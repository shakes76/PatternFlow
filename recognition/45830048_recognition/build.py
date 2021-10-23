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

X_train = X_train.unbatch()
a = tf.zeros([0, img_height, img_width, 1])
for image in X_train:
  a = tf.concat([a, [image]], axis = 0)
X_train = a

X_val = X_val.unbatch()
a = tf.zeros([0, img_height, img_width, 1])
for image in X_val:
  a = tf.concat([a, [image]], axis = 0)
X_val = a

y_train = y_train.unbatch()
a = tf.zeros([0, img_height, img_width, 1])
for image in y_train:
  a = tf.concat([a, [image]], axis = 0)
y_train = a

y_val = y_val.unbatch()
a = tf.zeros([0, img_height, img_width, 1])
for image in y_val:
  a = tf.concat([a, [image]], axis = 0)
y_val = a

print(y_train)

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(y_train.numpy()[i].astype("uint8"), cmap='gray')
  plt.axis("off")
plt.show()


plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(X_train.numpy()[i].astype("uint8"), cmap='gray')
  plt.axis("off")
plt.show()



model = improved_unet(img_height, img_width)
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(X_train, y_train, validation_data =(X_val, y_val), epochs = 30)

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