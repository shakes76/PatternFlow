import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow.keras.backend as K
import tensorflow_datasets as tfds
from tensorflow.keras import datasets, Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, concatenate, Concatenate, UpSampling2D, Conv2DTranspose, Dense
from PIL import Image
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import glob

# Load the TensorBoard notebook extension
%load_ext tensorboard

## Load functions

def load_image(img_path):
    return np.array(Image.open(fname)) / 255.

def load_image_one_hot(img_path):
    img =  np.array(Image.open(fname)) * (1./255.) # 0 to 1
    img = np.ndarray.astype(img, np.uint8)
    return (np.arange(img.max()+1) == img[...,None]).astype(np.uint8)


## Load datasets

data_dir = "./"
folders = ["ISIC2018_Task1-2_Training_Input_x2/",
          "ISIC2018_Task1_Training_GroundTruth_x2/"]
case_pattern = "ISIC_???????.jpg"
seg_pattern = "ISIC_???????_segmentation.png"

# Pre-defined
h, w = 1296, 1936
seed = 123

case_file_pattern = os.path.join(data_dir, folders[0], case_pattern)
seg_file_pattern = os.path.join(data_dir, folders[1], seg_pattern)

case_fname = glob.glob(case_file_pattern)
seg_fname = glob.glob(seg_file_pattern)

print("cases: ", len(case_fname))
print("segs: ", len(seg_fname))
print(case_file_pattern)
print(seg_file_pattern)
print(os.getcwd())

## DATA FORMATTING
train_frac = 0.85
case, seg = [], []

print("Reading case...")
for fname in (case_fname):
    case.append(load_image(fname))

# train_oh = load_image_one_hot(tf.constant(train_fname))

print("Reading seg...")
for fname in (seg_fname):
    seg.append(load_image_one_hot(fname))

train = np.array(train)
seg = np.array(seg)

case = case[:, :, :, np.newaxis]
seg = seg[:, :, :, np.newaxis]

print("Case images: ", case.shape, "\tdtype: ", case.dtype)
print("Seg images: ", seg.shape, "\tdtype: ", seg.dtype)

# del train_fname, test_fname, validate_fname


case_shuffler = np.random.permutation(len(case))
seg_shuffler = np.random.permutation(len(seg))

case = case[case_shuffler]
seg = seg[seg_shuffler]
