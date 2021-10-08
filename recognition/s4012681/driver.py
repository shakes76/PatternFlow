"""
Author: Richard Wainwright
Student ID: 40126812
Date: 05/10/2021

Driver for the UNet3d model for the classification of the Prostate 3D data set
"""

from unet import map_fn, get_nifti_data, one_hot
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
import nibabel
import skimage
from tensorflow.keras.utils import Sequence
import math


class MRIsequence(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]

        return np.array([
            get_nifti_data(file_name) for file_name in batch_x]), \
            ([one_hot(file_name) for file_name in batch_y])

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

IMG_WIDTH = 128
IMG_HEIGHT = 256
IMG_DEPTH = 256

mri_location = "/home/rick/3710data/semantic_MRs_anon/*.nii.gz"
label_location = "/home/rick/3710data/semantic_labels_anon/*nii.gz"

n_mri = len(glob.glob(mri_location))
n_labels = len(glob.glob(label_location))

mri_names = sorted(glob.glob(mri_location))
labels_names = sorted(glob.glob(label_location))

# split 15% of the files off as a test set
x_train, x_test, y_train, y_test = train_test_split(mri_names, labels_names,
                                                    test_size=0.15)

# split 15% of the training data off for validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.15)

train = MRIsequence(x_train, y_train, 8)
test = MRIsequence(x_test, y_test, 8)
val = MRIsequence(x_val, y_val, 8)

img = train[0][0][0]
mask = train[0][1][0]


print(type(img))

print(img.shape)
print(mask.shape)

print(np.unique(mask))
print(np.unique(img))

fig1, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(img[img.shape[0] // 2], cmap='gray')
ax2.imshow(mask[mask.shape[0] // 2], cmap='gray')
fig1.show()

# visualise all slices
fig2, ax1 = plt.subplots(1, 1, figsize=(13, 20))
ax1.imshow(skimage.util.montage(img))
ax1.set_title('image')
fig2.show()

fig3, ax1 = plt.subplots(1, 1, figsize=(13, 20))
ax1.imshow(skimage.util.montage(mask))
ax1.set_title('mask')
fig3.show()
