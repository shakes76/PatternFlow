import tensorflow as tf

import os
import math
import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image_dataset_from_directory

from IPython.display import display
from matplotlib import pyplot as plt


train_ds= image_dataset_from_directory(
    r"D:\Everything\Uni\2022\Sem2\COMP3710\report\AD_NC\train",
    image_size=(240, 256),
    label_mode=None,
    color_mode = "grayscale"
)

valid_ds = image_dataset_from_directory(
    r"D:\Everything\Uni\2022\Sem2\COMP3710\report\AD_NC\test",
    image_size=(240, 256),
    label_mode=None,
    color_mode = "grayscale"
)

        