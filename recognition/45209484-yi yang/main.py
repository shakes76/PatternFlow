'''
Pattern Recognition
Segment the ISICs data set with the UNet

@author Yi Yang
'''

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
import glob
from tensorflow.keras.layers import concatenate, Flatten
from tensorflow.keras.layers import Input, Dense, Conv2D, UpSampling2D, MaxPooling2D
from tensorflow.keras.models import Model

# load files
filelist_input = glob.glob("C:/Users/s4520948/Downloads/ISIC2018_Task1-2_Training_Input_x2/*.jpg")
filelist_ground_truth = glob.glob("C:/Users/s4520948/Downloads/ISIC2018_Task1_Training_GroundTruth_x2/*.png")
