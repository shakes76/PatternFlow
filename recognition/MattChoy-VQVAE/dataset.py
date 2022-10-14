"""
Setup:
1. Obtain a Personal Access Token from GitHub (requires GitHub Account)
    -> Navigate to https://github.com/settings/tokens to get a PAT
2. Clone ADNI dataset repo
    git clone https://{personal-access-token}@github.com/MattPChoy/ADNI-dataset.git data
"""

import tensorflow as tf
from constants import batch_size

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

train_ds = image_dataset_from_directory(
    os.path.join(dataset_fp, "train"),
    batch_size=batch_size,
    image_size=image_size,
    seed=dataset_seed,
    label_mode = None
)

test_ds = image_dataset_from_directory(
    os.path.join(dataset_fp, "test"),
    batch_size=batch_size,
    image_size=image_size,
    seed=dataset_seed,
    label_mode = None
)

def scaling(input_image):
    input_image = input_image / 255.0
    return input_image

train_ds = train_ds.map(scaling)
test_ds = test_ds.map(scaling)
