"""
Author: Richard Wainwright
Student ID: 40126812
Date: 05/10/2021

Driver for the UNet3d model for the classification of the Prostate 3D data set
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split

mri_location = "/home/rick/3710data/semantic_MRs_anon/*.nii.gz"
label_location = "/home/rick/3710data/semantic_labels_anon/*nii.gz"

n_mri = len(glob.glob(mri_location))
n_labels = len(glob.glob(label_location))

mri_names = sorted(glob.glob(mri_location))
labels_names = sorted(glob.glob(label_location))
print(n_mri, n_labels)

# split 15% of the files off as a test set
x_train, x_test, y_train, y_test = train_test_split(mri_names, labels_names, test_size=0.15)

# split 15% of the training data off for validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)

train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
test = tf.data.Dataset.from_tensor_slices((x_test, y_test))

