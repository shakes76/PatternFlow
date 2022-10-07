import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from dataset import *
from modules import *

val_split = 0.2
img_height = 256
img_width = 256
batch_size = 32

PATH = os.getcwd()
print(PATH)

train_path = PATH + "/ADNI_AD_NC_2D/AD_NC/train"
test_path = PATH + "/ADNI_AD_NC_2D/AD_NC/test"
train_ds = load_train_data(train_path, img_height, img_width, batch_size, val_split)
val_ds = load_validation_data(train_path, img_height, img_width, batch_size, val_split)
test_ds = load_test_data(test_path, img_height, img_width, batch_size)