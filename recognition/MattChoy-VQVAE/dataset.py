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
from constants import image_size, batch_size, dataset_seed, n_channels
import matplotlib.pyplot as plt
dataset_fp = os.path.join(os.getcwd(), "data", "ADNI")

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
    input_image = tf.image.rgb_to_grayscale(input_image)
    return input_image

train_ds = train_ds.map(scaling)
test_ds = test_ds.map(scaling)

for batch in train_ds:
    for im in batch:
        print(im.shape)
        plt.imshow(im)
        plt.show()
        break
    break


#
# n_batches = len(train_ds)
# n_samples = n_batches * batch_size
# ds_images = np.ndarray(shape=(len(train_ds) * batch_size, image_size[0], image_size[1], n_channels))
# idx=0
# for batch in train_ds:
#     for i in batch:
#         ds_images[idx] = i
#         idx += 1
#
# print(f"Dataset variance is {np.var(ds_images)/255.0}")
