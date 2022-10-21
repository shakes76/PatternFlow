# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:35:32 2020

@author: s4552215
"""
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images
from skimage.transform import resize
from skimage.morphology import label


import tensorflow as tf

from keras.models import Model, load_model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout
from keras.layers.core import Lambda, RepeatVector, Reshape
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalMaxPool2D
from keras.layers.merge import concatenate, add
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.utils import to_categorical
from keras import backend as K

import UNET_Model_Compile_ISIC_SM as UNC
import sys

# To Provide the input image and response image paths from the command line
img_path = str(sys.argv[1])
seg_path = str(sys.argv[2])
img_width = 256
img_height = 256

# Final model creation, compilation, prediction and calculation of Dice Score
UNC.mod_comp(img_path,seg_path,img_height,img_width)
