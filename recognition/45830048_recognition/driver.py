"""
Driver Script, trains model from model.py on ISIC2018 dataset

@author Lachlan Taylor
"""

import tensorflow as tf
from model import *
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K

# Global Variables
dir = "ISIC2018_Task1-2_Training_Input_x2"
mask_dir = "ISIC2018_Task1_Training_GroundTruth_x2"
batchs = 16
img_height = 256
img_width = 256

# Load image datasets as grayscale
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

X_train = X_train.unbatch()
X_val = X_val.unbatch()
y_train = y_train.unbatch()
y_val = y_val.unbatch()

#split off test data
test_size = int(0.2 * 2076)
X_test = X_train.take(test_size)
X_train = X_train.skip(test_size)
y_test = y_train.take(test_size)
y_train = y_train.skip(test_size)

# plot example images and masks
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

# normalise data and use as tensors instead of dataset type
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

# initialise, compile, train and evaluate model
model = improved_unet(img_height, img_width)
model.compile(optimizer = 'adam', loss = dice_coef_loss, metrics = ['accuracy', dice_coef])
history = model.fit(X_train, y_train, validation_data =(X_val, y_val), epochs = 30, batch_size = 16)

evaluation = model.evaluate(X_test, y_test)

print(evaluation)

predictions = model.predict(X_test)

# plot example results
fig, axs = plt.subplots(3, 3)
for i in range(3):
  temp = X_test.numpy()[i + 20] * 255
  axs[i, 0].imshow(temp.astype("uint8"), cmap='gray')
  axs[i, 0].axis("off")
  axs[i, 1].imshow(y_test.numpy()[i].astype("uint8"), cmap='gray')
  axs[i, 1].axis("off")
  axs[i, 2].imshow(predictions[i], cmap='gray')
  axs[i, 2].axis("off")
axs[0, 0].set_title("Testing Image")
axs[0, 1].set_title("Testing Ground Truth")
axs[0, 2].set_title("Generated Segment")
plt.show()

# plot performance
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