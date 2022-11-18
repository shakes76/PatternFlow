# Data Downloading
Documentation of old method of obtaining the dataset from source:

```python
"""
Dataset processing for the ADNI dataset. Downloaded from COMP3710 course site.
After downloading the .zip file, unzip it and delete unncessary folders.
The folder structure should be as follows:
    ./PatternFlow/recognition/MattChoy-VQVAE/data/ADNI/test/AD
    ./PatternFlow/recognition/MattChoy-VQVAE/data/ADNI/test/NC
    ./PatternFlow/recognition/MattChoy-VQVAE/data/ADNI/train/AD
    ./PatternFlow/recognition/MattChoy-VQVAE/data/ADNI/train/NC
In the test/ and train/ folders, the folder "AD" contains Alzheimer's class, and
                                 the folder "NC" contains Cognitive Normal samples.
"""
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

from util import check_adni_dataset, get_test_train_split

"""
Hyperparameters
"""
batch_size = 8
image_size = (256, 240) # Raw image size from dataset.
dataset_seed = 13

"""
Now load from the ./data directory
"""
dataset_fp = os.path.join(os.getcwd(), "data", "ADNI")
check_adni_dataset(dataset_fp)
get_test_train_split(dataset_fp)

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

# Perform scaling of the dataset - from [0, 255] to [0, 1]
def scaling(input_image):
    input_image = input_image / 255.0
    return input_image

train_ds = train_ds.map(scaling)
test_ds = test_ds.map(scaling)

```
