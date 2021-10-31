import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import keras
from tensorflow import keras
from keras import layers, preprocessing
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, LeakyReLU, Dropout, BatchNormalization, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.python.keras.layers.convolutional import Conv2DTranspose
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()