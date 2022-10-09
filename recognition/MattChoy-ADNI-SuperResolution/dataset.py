"""
Dataset processing for the ADNI dataset. Downloaded from COMP3710 course site.
After downloading the .zip file, unzip it and delete unncessary folders.
The folder structure should be as follows:
    ./PatternFlow/recognition/MattChoy-ADNI-SuperResolution/data/ADNI/test/AD
    ./PatternFlow/recognition/MattChoy-ADNI-SuperResolution/data/ADNI/test/NC
    ./PatternFlow/recognition/MattChoy-ADNI-SuperResolution/data/ADNI/train/AD
    ./PatternFlow/recognition/MattChoy-ADNI-SuperResolution/data/ADNI/train/NC
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

