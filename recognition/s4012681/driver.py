"""
Author: Richard Wainwright
Student ID: 40126812
Date: 05/10/2021

Driver for the U-net3d model for the classification of the Prostate 3D data set.
Presented as a script for ease of use.  Most changes can be made to the factors
listed at the top of this file.
"""

from unet import get_image, get_mask, unet, reshape, dice, plt_compare
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.keras.utils import Sequence
import math


# Shape of the MRIs
IMG_WIDTH = 128
IMG_HEIGHT = 256
IMG_DEPTH = 256
IMG_CHANNELS = 1

# Hyper-parameters for training
BATCH_SIZE = 1
FILTERS = 8
EPOCHS = 50

# Image location, change to match saved image locations
train_mri_location = "./semantic_MRs_anon/train/*.nii.gz"
train_label_location = "./semantic_labels_anon/train/*nii.gz"

val_mri_location = "./semantic_MRs_anon/val/*.nii.gz"
val_label_location = "./semantic_labels_anon/val/*nii.gz"

test_mri_location = "./semantic_MRs_anon/test/*.nii.gz"
test_label_location = "./semantic_labels_anon/test/*nii.gz"


class MRISequence(Sequence):
    """
    The generator to provide (image, mask). Shuffles order after each epoch
    Based on example code at:
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
    """
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.indices = list(range(len(self.x)))

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        return np.array([reshape(1, (get_image(file_name))) for file_name in batch_x]), \
               np.array([reshape(6, (get_mask(file_name))) for file_name in batch_y])

    def on_epoch_end(self):
        np.random.shuffle(self.indices)


class CustomCallback(tf.keras.callbacks.Callback):
    """
    Used to save a prediction after each epoch.  Not currently used or required
    by the model but useful in assessing performance when adjusting parameters
    or adding functionality, to use add to model.fit(callbacks=[CustomCallback()])
    """
    def on_epoch_end(self, epoch):
        pred = model.predict(test)
        mask = pred[0]
        mask = tf.math.argmax(mask, axis=-1)
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(mask[mask.shape[0] // 2], cmap='gray')
        fig.savefig("simple{}.png".format(epoch))


# Get all file names of MRIs and labels
train_mri_names = sorted(glob.glob(train_mri_location))
train_labels_names = sorted(glob.glob(train_label_location))
val_mri_names = sorted(glob.glob(val_mri_location))
val_labels_names = sorted(glob.glob(val_label_location))
test_mri_names = sorted(glob.glob(test_mri_location))
test_labels_names = sorted(glob.glob(test_label_location))

# Create Sequences to generate pairs of MRI and mask
train = MRISequence(train_mri_names, train_labels_names, BATCH_SIZE)
val = MRISequence(val_mri_names, val_labels_names, BATCH_SIZE)
test = MRISequence(test_mri_names, test_labels_names, BATCH_SIZE)

# Create and compile model
model = unet(FILTERS)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary(line_length=120)

# Fit the training data and store for the evaluation plot
curves = model.fit(train, epochs=EPOCHS, validation_data=val, batch_size=BATCH_SIZE)

# Evaluate the model with the test data and generate predictions
print("Evaluation:")
model.evaluate(test)
predictions = model.predict(test)

# Separate the test images and labels for Dice calculation and printing
test_images = np.empty((len(test), IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, 1), dtype=np.float32)
test_labels = np.empty((len(test), IMG_WIDTH, IMG_HEIGHT, IMG_DEPTH, 6), dtype=np.float32)
for i in range(len(test)):
    test_images[i] = test[i][0]
    test_labels[i] = test[i][1]

# Report Dice similarity coefficients for each category
print("Body: ", dice(test_labels[..., 1], predictions[..., 1]))
print("Bones: ", dice(test_labels[..., 2], predictions[..., 2]))
print("Bladder: ", dice(test_labels[..., 3], predictions[..., 3]))
print("Rectum: ", dice(test_labels[..., 4], predictions[..., 4]))
print("Prostate: ", dice(test_labels[..., 5], predictions[..., 5]))
print("Overall (including background): ", dice(test_labels, predictions))

# Plot the model performance
# plot accuracy
fig2, (gax1, gax2) = plt.subplots(1, 2)
gax1.plot(curves.history['accuracy'])
gax1.plot(curves.history['val_accuracy'])
gax1.legend(['train', 'test'], loc='upper left')
gax1.title.set_text("Accuracy")
# plot loss
gax2.plot(curves.history['loss'])
gax2.plot(curves.history['val_loss'])
gax2.legend(['train', 'test'], loc='upper left')
gax2.title.set_text("Loss")
fig2.savefig('acc_loss.png')

# Save images comparing MRI, mask and prediction from the test set
for i in range(len(test)):
    plt_compare(test_images[i], test_labels[i], predictions[i], i)
