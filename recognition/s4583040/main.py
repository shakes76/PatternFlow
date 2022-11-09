"""
The main test driver script that runs the improved UNet algorithm.

Author: Siwan Li
Student ID: s4583040
Date: 1 November 2021
GitHub Name: Kirbologist
"""

from preprocessing import load_data, split_train_test_val
from UNet_model import Improved_UNet_Model, dice_coef, dice_coef_loss
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import matplotlib.pyplot as plt

# Directories where image and masks are located. Please change this for your files
mask_path = '../ISIC2018_Task1_Training_GroundTruth_x2'
img_path = '../ISIC2018_Task1-2_Training_Input_x2'

# Number of (image, mask) pairs
size = 2594

# Data preprocessing and splitting
image_ds = load_data(mask_path, img_path)
train_image, val_image, test_image = split_train_test_val(image_ds, size)

# Splitting data into batches
train_size = len(list(train_image))
val_size = len(list(val_image))
test_size = len(list(test_image))

BATCH_SIZE=32
STEPS_PER_EPOCH =train_size//BATCH_SIZE
train_image = train_image.batch(BATCH_SIZE).repeat()
val_image = val_image.batch(BATCH_SIZE)
test_image = test_image.batch(1)

# Building model and training parameters
model = Improved_UNet_Model()

opt = SGD(lr=0.2)

initial_learning_rate = 0.005
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.985,
    staircase=True)
opt = SGD(learning_rate=lr_schedule)


model.compile(optimizer=opt, loss=dice_coef_loss, metrics=[dice_coef])

VALIDATION_STEPS = val_size//BATCH_SIZE

# Training model
model_history = model.fit(train_image,steps_per_epoch=STEPS_PER_EPOCH ,epochs=250, validation_data=val_image)


# Plotting the loss and dice coefficient of the model over time
plt.figure(1)
plt.subplot(2, 1, 1)
plt.plot(range(len(model_history.history['loss'])),model_history.history['loss'], label='loss')
plt.plot(range(len(model_history.history['val_loss'])),model_history.history['val_loss'], label='val_loss')
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(range(len(model_history.history['dice_coef'])),model_history.history['dice_coef'], label='dice_coef')
plt.plot(range(len(model_history.history['val_dice_coef'])),model_history.history['val_dice_coef'], label='val_dice_coef')
plt.legend()
plt.show()
