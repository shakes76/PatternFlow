"""
COMP3170
Jialiang Hou
45996216
training and save the model
"""
import dataset
import modules
from keras.models import load_model
import random
import tensorflow as tf
import tensorflow
import os
import tensorflow.keras.backend as K
import numpy as np


# get the model from modules.py
model = modules.SiameseNetwork()
# get the training and test sample from dataset.py
train_x1, train_x2, train_y, test_x1, test_x2, test_y = dataset.get_dataset()

# fit
model.fit([train_x1, train_x2], train_y, epochs=20, batch_size=16)
model.save('my_model.h5')

