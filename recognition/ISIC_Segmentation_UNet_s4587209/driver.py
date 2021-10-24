import glob
import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Constants
ISIC2018_data_link = "https://cloudstor.aarnet.edu.au/sender/download.php?token=b66a9288-2f00-4330-82ea-9b8711d27643&files_ids=14200406"
download_directory = os.getcwd() + '\ISIC2018_Task1-2_Training_Data.zip'

print(download_directory)

# Downloads files if not present
tf.keras.utils.get_file(origin=ISIC2018_data_link, fname=download_directory, extract=True, cache_dir=os.getcwd())
# Segments folders into arrays
inputs = glob.glob('datasets/ISIC2018_Task1-2_Training_Input_x2/*.jpg')
labels = glob.glob('datasets/ISIC2018_Task1_Training_GroundTruth_x2/*.png')
