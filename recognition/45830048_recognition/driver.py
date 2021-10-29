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
#dir = "C:/ISIC_IMAGE_TEST"
#mask_dir = "C:/ISIC_MASK_TEST"
batchs = 16
img_height = 256
img_width = 256

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

#2076
test_size = int(0.2 * 84)
X_test = X_train.take(test_size)
X_train = X_train.skip(test_size)
y_test = y_train.take(test_size)
y_train = y_train.skip(test_size)


plt.figure(figsize=(10, 10))
i = 0
for images in X_train:
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(images.numpy().astype("uint8"), cmap='gray')
  plt.axis("off")
  i += 1
  if i == 9:
    break
plt.show()
#plt.savefig("/PatternFlow/recognition/45830048_recognition/train_example.png")


plt.figure(figsize=(10, 10))
i = 0
for images in y_train:
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(images.numpy().astype("uint8"), cmap='gray')
  plt.axis("off")
  i += 1
  if i == 9:
    break
plt.show()
#plt.savefig("/PatternFlow/recognition/45830048_recognition/train_example.png")


#plt.figure(figsize=(10, 10))
#for i in range(9):
#  ax = plt.subplot(3, 3, i + 1)
#  plt.imshow(y_train.take(1)[i].numpy().astype("uint8"), cmap='gray')
#  plt.axis("off")
##plt.show()
#plt.savefig("ground_truth_example")
#
#
#
#
#plt.figure(figsize=(10, 10))
#for i in range(9):
#  ax = plt.subplot(3, 3, i + 1)
#  plt.imshow(X_train.numpy()[i].astype("uint8"), cmap='gray')
#  plt.axis("off")
##plt.show()
#plt.savefig("train_example")

print("normalising")
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

print("normalising complete")

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
history = model.fit(X_train, y_train, validation_data =(X_val, y_val), epochs = 30, batch_size = 16)

evaluation = model.evaluate(X_test, y_test)

print(evaluation)

predictions = model.predict(X_test)

print("done")
fig, axs = plt.subplots(3, 3)
for i in range(3):
  #print(i)
  #x = tf.gather(X_test, i + 2)
  #y = tf.gather(y_test, i + 2)
  #print(x)
  #print(y)
  #prediction = model.predict(x)
  #predictions.numpy()[i] *= 255
  #ax = plt.subplot(3, 3, 1)
  temp = X_test.numpy()[i] * 255
  axs[i, 0].imshow(temp.astype("uint8"), cmap='gray')
  axs[i, 0].axis("off")
  #ax = plt.subplot(3, 3, 2)
  axs[i, 1].imshow(y_test.numpy()[i].astype("uint8"), cmap='gray')
  axs[i, 1].axis("off")
  #ax = plt.subplot(3, 3, 3)
  axs[i, 2].imshow(predictions[i], cmap='gray')
  axs[i, 2].axis("off")


axs[0, 0].set_title("Testing Image")
axs[0, 1].set_title("Testing Ground Truth")
axs[0, 2].set_title("Generated Segment")
plt.show()

plt.title("Accuracy each Epoch")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.plot(history.history['accuracy'], label="train accuracy")
plt.plot(history.history['val_accuracy'], label="val accuracy")
plt.legend()
plt.show()
plt.title("Dice Coeffecient each Epoch")
plt.ylabel("Dice Coeffecient")
plt.xlabel("Epoch")
plt.plot(history.history['dice_coef'], label="train dice coeffecient")
plt.plot(history.history['val_dice_coef'], label="val dice coeffecient")
plt.legend()
plt.show()


#plt.savefig("result_example")

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